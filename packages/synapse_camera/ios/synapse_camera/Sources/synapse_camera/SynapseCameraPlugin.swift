import AVFoundation
import CoreMedia
import Flutter
import UIKit

/// AVFoundation implementation of the Synapse clinical camera.
///
/// * Selects a 420f (full-range YUV) format that sustains the target frame
///   rate (120 fps preferred, 60 fps floor) and pins the frame duration.
/// * `lock` switches exposure to `.custom` (fixed ISO + duration), focus to
///   `.locked` at a fixed lens position and white balance to `.locked`, and
///   disables HDR, low-light boost and stabilization so nothing adapts mid-scan.
/// * The Y plane *is* the grayscale image, so frames are cropped straight from
///   it – no colour conversion is needed.
/// * Frame and torch timestamps are expressed on the mach host clock, so the
///   stimulus and the response share one time base.
public final class SynapseCameraPlugin: NSObject, FlutterPlugin, FlutterStreamHandler, FlutterTexture,
  AVCaptureVideoDataOutputSampleBufferDelegate
{
  public static func register(with registrar: FlutterPluginRegistrar) {
    let instance = SynapseCameraPlugin(textures: registrar.textures())
    let methods = FlutterMethodChannel(name: "synapse/camera", binaryMessenger: registrar.messenger())
    registrar.addMethodCallDelegate(instance, channel: methods)
    let events = FlutterEventChannel(name: "synapse/camera/frames", binaryMessenger: registrar.messenger())
    events.setStreamHandler(instance)
  }

  private let textures: FlutterTextureRegistry
  private var textureId: Int64 = -1

  private let session = AVCaptureSession()
  private let output = AVCaptureVideoDataOutput()
  private var device: AVCaptureDevice?
  private var fps: Int32 = 120

  private let sessionQueue = DispatchQueue(label: "synapse.camera.session")
  private let videoQueue = DispatchQueue(label: "synapse.camera.video", qos: .userInteractive)
  private let torchQueue = DispatchQueue(label: "synapse.camera.torch", qos: .userInteractive)

  private let stateLock = NSLock()
  private var latestPixelBuffer: CVPixelBuffer?
  private var eventSink: FlutterEventSink?
  private var roiSize = 256
  private var frameIndex: Int64 = 0
  private var torchOn = false
  private var pulseTimer: DispatchSourceTimer?

  init(textures: FlutterTextureRegistry) {
    self.textures = textures
    super.init()
  }

  // MARK: - Method channel

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    let args = call.arguments as? [String: Any] ?? [:]
    switch call.method {
    case "initialize":
      initialize(args: args, result: result)
    case "lock":
      lock(args: args, result: result)
    case "unlock":
      unlock(result: result)
    case "flashPulse":
      flashPulse(
        durationMs: args["durationMs"] as? Int ?? 100,
        level: Float(args["level"] as? Double ?? 1.0),
        result: result)
    case "nowUs":
      result(Self.hostTimeUs())
    case "dispose":
      dispose(result: result)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  private func initialize(args: [String: Any], result: @escaping FlutterResult) {
    let targetFps = args["targetFps"] as? Int ?? 120
    let minFps = args["minFps"] as? Int ?? 60
    roiSize = args["roiSize"] as? Int ?? 256

    AVCaptureDevice.requestAccess(for: .video) { granted in
      guard granted else {
        result(FlutterError(code: "permission", message: "Camera permission denied", details: nil))
        return
      }
      self.sessionQueue.async {
        do {
          let reply = try self.configureSession(targetFps: targetFps, minFps: minFps)
          DispatchQueue.main.async { result(reply) }
        } catch let error as FlutterError {
          DispatchQueue.main.async { result(error) }
        } catch {
          DispatchQueue.main.async {
            result(FlutterError(code: "camera", message: error.localizedDescription, details: nil))
          }
        }
      }
    }
  }

  private func configureSession(targetFps: Int, minFps: Int) throws -> [String: Any] {
    guard
      let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
      device.hasTorch
    else {
      throw FlutterError(code: "hardware", message: "Rear camera with torch is required", details: nil)
    }

    guard let (format, achievedFps) = Self.bestFormat(for: device, targetFps: targetFps, minFps: minFps)
    else {
      throw FlutterError(
        code: "fps", message: "No camera format sustains \(minFps) fps", details: nil)
    }

    session.beginConfiguration()
    session.sessionPreset = .inputPriority
    session.inputs.forEach { session.removeInput($0) }
    session.outputs.forEach { session.removeOutput($0) }

    let input = try AVCaptureDeviceInput(device: device)
    guard session.canAddInput(input) else {
      session.commitConfiguration()
      throw FlutterError(code: "camera", message: "Cannot add camera input", details: nil)
    }
    session.addInput(input)

    output.videoSettings = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
    ]
    // Never let AVFoundation drop frames silently because we were slow: the
    // Dart side measures inter-frame gaps and flags them explicitly instead.
    output.alwaysDiscardsLateVideoFrames = false
    output.setSampleBufferDelegate(self, queue: videoQueue)
    guard session.canAddOutput(output) else {
      session.commitConfiguration()
      throw FlutterError(code: "camera", message: "Cannot add video output", details: nil)
    }
    session.addOutput(output)

    if let connection = output.connection(with: .video) {
      if connection.isVideoStabilizationSupported {
        connection.preferredVideoStabilizationMode = .off
      }
      if #available(iOS 17.0, *) {
        if connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
      } else if connection.isVideoOrientationSupported {
        connection.videoOrientation = .portrait
      }
    }

    try device.lockForConfiguration()
    device.activeFormat = format
    let frameDuration = CMTime(value: 1, timescale: CMTimeScale(achievedFps))
    device.activeVideoMinFrameDuration = frameDuration
    device.activeVideoMaxFrameDuration = frameDuration
    if device.isLowLightBoostSupported { device.automaticallyEnablesLowLightBoostWhenAvailable = false }
    if format.isVideoHDRSupported {
      device.automaticallyAdjustsVideoHDREnabled = false
      device.isVideoHDREnabled = false
    }
    device.isSubjectAreaChangeMonitoringEnabled = false
    // Converge AE/AF on the eye at the center before the clinician taps.
    let center = CGPoint(x: 0.5, y: 0.5)
    if device.isFocusPointOfInterestSupported { device.focusPointOfInterest = center }
    if device.isFocusModeSupported(.continuousAutoFocus) { device.focusMode = .continuousAutoFocus }
    if device.isExposurePointOfInterestSupported { device.exposurePointOfInterest = center }
    if device.isExposureModeSupported(.continuousAutoExposure) {
      device.exposureMode = .continuousAutoExposure
    }
    device.unlockForConfiguration()
    session.commitConfiguration()

    self.device = device
    self.fps = Int32(achievedFps)
    if textureId < 0 { textureId = textures.register(self) }
    session.startRunning()

    let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
    return [
      "textureId": textureId,
      // Buffers are rotated to portrait by the connection.
      "previewWidth": Int(min(dims.width, dims.height)),
      "previewHeight": Int(max(dims.width, dims.height)),
      "fps": achievedFps,
      "quarterTurns": 0,
      "manualSensor": true,
    ]
  }

  /// Prefers the target fps, then the resolution closest to 1080p (never
  /// above it – larger buffers only cost bandwidth), full-range 420 only.
  private static func bestFormat(for device: AVCaptureDevice, targetFps: Int, minFps: Int)
    -> (AVCaptureDevice.Format, Int)?
  {
    var best: (format: AVCaptureDevice.Format, fps: Int, score: Double)?
    for format in device.formats {
      guard
        CMFormatDescriptionGetMediaSubType(format.formatDescription)
          == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
      else { continue }
      let maxRate = format.videoSupportedFrameRateRanges.map { $0.maxFrameRate }.max() ?? 0
      let usable = Int(min(maxRate, Double(targetFps)).rounded(.down))
      guard usable >= minFps else { continue }
      let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
      let pixels = Double(dims.width) * Double(dims.height)
      guard pixels <= 1920 * 1080 else { continue }
      let score = Double(usable) * 1e7 - abs(pixels - 1920 * 1080)
      if best == nil || score > best!.score { best = (format, usable, score) }
    }
    return best.map { ($0.format, $0.fps) }
  }

  private func lock(args: [String: Any], result: @escaping FlutterResult) {
    sessionQueue.async {
      guard let device = self.device else {
        DispatchQueue.main.async {
          result(FlutterError(code: "state", message: "Camera not initialized", details: nil))
        }
        return
      }
      do {
        try device.lockForConfiguration()
        let format = device.activeFormat
        let iso = Self.clamp(
          Float(args["iso"] as? Double ?? Double(device.iso)), format.minISO, format.maxISO)
        let framePeriod = CMTime(value: 1, timescale: CMTimeScale(self.fps))
        var duration = (args["exposureUs"] as? Double).map {
          CMTime(seconds: $0 / 1e6, preferredTimescale: 1_000_000)
        } ?? device.exposureDuration
        if CMTimeCompare(duration, framePeriod) > 0 { duration = framePeriod }
        if CMTimeCompare(duration, format.minExposureDuration) < 0 {
          duration = format.minExposureDuration
        }
        let lens = Self.clamp(Float(args["focus"] as? Double ?? Double(device.lensPosition)), 0, 1)
        if device.isFocusModeSupported(.locked) {
          device.setFocusModeLocked(lensPosition: lens, completionHandler: nil)
        }
        if device.isWhiteBalanceModeSupported(.locked) { device.whiteBalanceMode = .locked }
        device.setExposureModeCustom(duration: duration, iso: iso) { _ in
          DispatchQueue.main.async {
            result([
              "iso": Double(iso),
              "exposureUs": duration.seconds * 1e6,
              "focus": Double(lens),
              "fps": Double(self.fps),
              "manual": true,
            ])
          }
        }
        device.unlockForConfiguration()
      } catch {
        DispatchQueue.main.async {
          result(FlutterError(code: "lock", message: error.localizedDescription, details: nil))
        }
      }
    }
  }

  private func unlock(result: @escaping FlutterResult) {
    sessionQueue.async {
      if let device = self.device, (try? device.lockForConfiguration()) != nil {
        if device.isExposureModeSupported(.continuousAutoExposure) {
          device.exposureMode = .continuousAutoExposure
        }
        if device.isFocusModeSupported(.continuousAutoFocus) {
          device.focusMode = .continuousAutoFocus
        }
        if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
          device.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        device.unlockForConfiguration()
      }
      DispatchQueue.main.async { result(nil) }
    }
  }

  /// Torch pulse driven by a strict, zero-leeway GCD timer on a
  /// user-interactive queue. Both edges are timestamped on the host clock.
  private func flashPulse(durationMs: Int, level: Float, result: @escaping FlutterResult) {
    torchQueue.async {
      guard let device = self.device, device.hasTorch else {
        DispatchQueue.main.async {
          result(FlutterError(code: "hardware", message: "Torch unavailable", details: nil))
        }
        return
      }
      do {
        try device.lockForConfiguration()
        let onUs = Self.hostTimeUs()
        try device.setTorchModeOn(level: Self.clamp(level, 0.01, AVCaptureDevice.maxAvailableTorchLevel))
        device.unlockForConfiguration()
        self.setTorchFlag(true)

        let timer = DispatchSource.makeTimerSource(flags: .strict, queue: self.torchQueue)
        let fire = DispatchTime(uptimeNanoseconds: UInt64(onUs) * 1000 + UInt64(durationMs) * 1_000_000)
        timer.schedule(deadline: fire, leeway: .nanoseconds(0))
        timer.setEventHandler {
          let offUs = Self.hostTimeUs()
          if (try? device.lockForConfiguration()) != nil {
            device.torchMode = .off
            device.unlockForConfiguration()
          }
          self.setTorchFlag(false)
          self.pulseTimer?.cancel()
          self.pulseTimer = nil
          DispatchQueue.main.async { result(["onUs": onUs, "offUs": offUs]) }
        }
        self.pulseTimer = timer
        timer.resume()
      } catch {
        DispatchQueue.main.async {
          result(FlutterError(code: "torch", message: error.localizedDescription, details: nil))
        }
      }
    }
  }

  private func dispose(result: @escaping FlutterResult) {
    sessionQueue.async {
      self.session.stopRunning()
      if let device = self.device, device.hasTorch, (try? device.lockForConfiguration()) != nil {
        device.torchMode = .off
        device.unlockForConfiguration()
      }
      DispatchQueue.main.async {
        if self.textureId >= 0 {
          self.textures.unregisterTexture(self.textureId)
          self.textureId = -1
        }
        self.stateLock.lock()
        self.latestPixelBuffer = nil
        self.stateLock.unlock()
        result(nil)
      }
    }
  }

  // MARK: - Frames

  public func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection
  ) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

    stateLock.lock()
    latestPixelBuffer = pixelBuffer
    let sink = eventSink
    let torch = torchOn
    let side = roiSize
    frameIndex += 1
    let index = frameIndex
    stateLock.unlock()
    textures.textureFrameAvailable(textureId)

    guard let sink = sink else { return }

    let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
    let hostPts: CMTime
    if #available(iOS 15.4, *), let clock = session.synchronizationClock {
      hostPts = CMSyncConvertTime(pts, from: clock, to: CMClockGetHostTimeClock())
    } else {
      hostPts = pts
    }

    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return }
    let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
    let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
    let stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
    let s = min(side, width, height)
    let x0 = (width - s) / 2
    let y0 = (height - s) / 2
    var roi = Data(count: s * s)
    roi.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
      let src = base.assumingMemoryBound(to: UInt8.self)
      for row in 0..<s {
        memcpy(dst.baseAddress! + row * s, src + (y0 + row) * stride + x0, s)
      }
    }

    let event: [String: Any] = [
      "w": s, "h": s,
      "ts": Int64(hostPts.seconds * 1e6),
      "idx": index,
      "torch": torch,
      "bytes": FlutterStandardTypedData(bytes: roi),
    ]
    DispatchQueue.main.async { sink(event) }
  }

  public func copyPixelBuffer() -> Unmanaged<CVPixelBuffer>? {
    stateLock.lock()
    defer { stateLock.unlock() }
    guard let buffer = latestPixelBuffer else { return nil }
    return Unmanaged.passRetained(buffer)
  }

  public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink)
    -> FlutterError?
  {
    stateLock.lock()
    eventSink = events
    stateLock.unlock()
    return nil
  }

  public func onCancel(withArguments arguments: Any?) -> FlutterError? {
    stateLock.lock()
    eventSink = nil
    stateLock.unlock()
    return nil
  }

  // MARK: - Helpers

  private func setTorchFlag(_ on: Bool) {
    stateLock.lock()
    torchOn = on
    stateLock.unlock()
  }

  /// Microseconds on the mach host clock (same clock as DispatchTime).
  static func hostTimeUs() -> Int64 {
    Int64(DispatchTime.now().uptimeNanoseconds / 1000)
  }

  private static func clamp<T: Comparable>(_ v: T, _ lo: T, _ hi: T) -> T { min(max(v, lo), hi) }
}
