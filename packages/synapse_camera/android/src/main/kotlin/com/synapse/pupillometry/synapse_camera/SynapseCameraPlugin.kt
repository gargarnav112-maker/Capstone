package com.synapse.pupillometry.synapse_camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.util.Range
import android.util.Size
import android.view.Surface
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.view.TextureRegistry
import java.util.concurrent.Executor
import kotlin.math.abs
import kotlin.math.min

/**
 * Camera2 implementation of the Synapse clinical camera.
 *
 * Locking uses CONTROL_AE_MODE_OFF / CONTROL_AF_MODE_OFF with absolute
 * SENSOR_SENSITIVITY (ISO), SENSOR_EXPOSURE_TIME and LENS_FOCUS_DISTANCE when
 * the device advertises MANUAL_SENSOR; otherwise it falls back to AE/AWB lock.
 * The torch is driven through FLASH_MODE_TORCH on the repeating request so it
 * shares the capture pipeline (and its timestamps) with the frames.
 */
class SynapseCameraPlugin : FlutterPlugin, MethodChannel.MethodCallHandler, EventChannel.StreamHandler {
  private lateinit var context: Context
  private lateinit var textures: TextureRegistry
  private lateinit var methods: MethodChannel
  private lateinit var events: EventChannel
  private val main = Handler(Looper.getMainLooper())

  private var cameraThread: HandlerThread? = null
  private var cameraHandler: Handler? = null
  private var torchThread: HandlerThread? = null
  private var torchHandler: Handler? = null

  private var device: CameraDevice? = null
  private var session: CameraCaptureSession? = null
  private var reader: ImageReader? = null
  private var textureEntry: TextureRegistry.SurfaceTextureEntry? = null
  private var previewSurface: Surface? = null
  private var characteristics: CameraCharacteristics? = null

  private var fpsRange = Range(120, 120)
  private var roiSize = 256
  private var sensorOrientation = 90
  private var manualSensor = false
  private var realtimeTimestamps = false

  @Volatile private var sink: EventChannel.EventSink? = null
  @Volatile private var torchOn = false
  @Volatile private var lastIso = 100
  @Volatile private var lastExposureNs = 8_000_000L
  @Volatile private var lastFocus = 0f
  private var frameIndex = 0L

  // The repeating request is rebuilt from this state whenever it changes.
  private var locked = false
  private var lockIso = 100
  private var lockExposureNs = 8_000_000L
  private var lockFocus = 0f

  override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    context = binding.applicationContext
    textures = binding.textureRegistry
    methods = MethodChannel(binding.binaryMessenger, "synapse/camera")
    methods.setMethodCallHandler(this)
    events = EventChannel(binding.binaryMessenger, "synapse/camera/frames")
    events.setStreamHandler(this)
  }

  override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
    methods.setMethodCallHandler(null)
    events.setStreamHandler(null)
    release()
  }

  override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
    try {
      when (call.method) {
        "initialize" -> initialize(
          call.argument<Int>("targetFps") ?: 120,
          call.argument<Int>("minFps") ?: 60,
          call.argument<Int>("roiSize") ?: 256,
          result,
        )
        "lock" -> lock(
          call.argument<Double>("iso"),
          call.argument<Double>("exposureUs"),
          call.argument<Double>("focus"),
          result,
        )
        "unlock" -> { locked = false; applyRepeating(); result.success(null) }
        "flashPulse" -> flashPulse(call.argument<Int>("durationMs") ?: 100, result)
        "nowUs" -> result.success(nowNs() / 1000)
        "dispose" -> { release(); result.success(null) }
        else -> result.notImplemented()
      }
    } catch (e: Exception) {
      result.error("camera", e.message, null)
    }
  }

  // ---------------------------------------------------------------- setup

  @SuppressLint("MissingPermission")
  private fun initialize(targetFps: Int, minFps: Int, roi: Int, result: MethodChannel.Result) {
    if (context.checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
      result.error("permission", "Camera permission not granted", null)
      return
    }
    release()
    roiSize = roi
    cameraThread = HandlerThread("synapse-camera", Process.THREAD_PRIORITY_URGENT_DISPLAY).also { it.start() }
    cameraHandler = Handler(cameraThread!!.looper)
    torchThread = HandlerThread("synapse-torch", Process.THREAD_PRIORITY_URGENT_DISPLAY).also { it.start() }
    torchHandler = Handler(torchThread!!.looper)

    val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    val id = manager.cameraIdList.firstOrNull { cid ->
      val c = manager.getCameraCharacteristics(cid)
      c.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK &&
        c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
    } ?: run {
      result.error("hardware", "Rear camera with torch is required", null)
      return
    }
    val chars = manager.getCameraCharacteristics(id)
    characteristics = chars
    sensorOrientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
    manualSensor = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
      ?.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
    realtimeTimestamps = chars.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE) ==
      CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME

    // Highest fixed (lower == upper) AE fps range, capped at the target.
    val ranges = chars.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES) ?: emptyArray()
    val range = ranges
      .filter { it.upper in minFps..targetFps }
      .maxWithOrNull(compareBy<Range<Int>>({ it.upper }, { it.lower }))
      ?: run {
        result.error("fps", "No AE range sustains $minFps fps", null)
        return
      }
    fpsRange = Range(range.upper, range.upper).takeIf { r -> ranges.any { it == r } } ?: range

    val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
    val frameNs = 1_000_000_000L / fpsRange.upper
    val size = map.getOutputSizes(ImageFormat.YUV_420_888)
      .filter { it.width * it.height <= 1920 * 1080 }
      .filter { map.getOutputMinFrameDuration(ImageFormat.YUV_420_888, it) <= frameNs }
      .minByOrNull { abs(it.width * it.height - 1280 * 720) }
      ?: Size(1280, 720)

    val entry = textures.createSurfaceTexture()
    textureEntry = entry
    entry.surfaceTexture().setDefaultBufferSize(size.width, size.height)
    previewSurface = Surface(entry.surfaceTexture())
    reader = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, 4).also {
      it.setOnImageAvailableListener({ r -> onImage(r) }, cameraHandler)
    }

    manager.openCamera(id, object : CameraDevice.StateCallback() {
      override fun onOpened(camera: CameraDevice) {
        device = camera
        createSession(camera, size, result)
      }
      override fun onDisconnected(camera: CameraDevice) { camera.close(); device = null }
      override fun onError(camera: CameraDevice, error: Int) {
        camera.close(); device = null
        main.post { result.error("camera", "Camera error $error", null) }
      }
    }, cameraHandler)
  }

  private fun createSession(camera: CameraDevice, size: Size, result: MethodChannel.Result) {
    val outputs = listOf(OutputConfiguration(previewSurface!!), OutputConfiguration(reader!!.surface))
    val executor = Executor { cameraHandler!!.post(it) }
    val callback = object : CameraCaptureSession.StateCallback() {
      override fun onConfigured(s: CameraCaptureSession) {
        session = s
        applyRepeating()
        main.post {
          result.success(
            mapOf(
              "textureId" to textureEntry!!.id(),
              "previewWidth" to size.width,
              "previewHeight" to size.height,
              "fps" to fpsRange.upper.toDouble(),
              "quarterTurns" to sensorOrientation / 90,
              "manualSensor" to manualSensor,
            ),
          )
        }
      }
      override fun onConfigureFailed(s: CameraCaptureSession) {
        main.post { result.error("camera", "Session configuration failed", null) }
      }
    }
    camera.createCaptureSession(
      SessionConfiguration(SessionConfiguration.SESSION_REGULAR, outputs, executor, callback),
    )
  }

  /** Rebuilds the repeating request from the current lock/torch state. */
  private fun applyRepeating() {
    val camera = device ?: return
    val s = session ?: return
    val b = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD)
    b.addTarget(previewSurface!!)
    b.addTarget(reader!!.surface)
    b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, fpsRange)
    b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF)
    b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE, CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF)

    if (locked && manualSensor) {
      b.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
      b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
      b.set(CaptureRequest.SENSOR_SENSITIVITY, lockIso)
      b.set(CaptureRequest.SENSOR_EXPOSURE_TIME, lockExposureNs)
      b.set(CaptureRequest.SENSOR_FRAME_DURATION, 1_000_000_000L / fpsRange.upper)
      b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
      b.set(CaptureRequest.LENS_FOCUS_DISTANCE, lockFocus)
      b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
    } else if (locked) {
      b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
      b.set(CaptureRequest.CONTROL_AE_LOCK, true)
      b.set(CaptureRequest.CONTROL_AWB_LOCK, true)
      b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
    } else {
      b.set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
      b.set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
    }
    // FLASH_MODE is honoured when AE mode is ON or OFF (not the auto-flash modes).
    b.set(
      CaptureRequest.FLASH_MODE,
      if (torchOn) CameraMetadata.FLASH_MODE_TORCH else CameraMetadata.FLASH_MODE_OFF,
    )
    s.setRepeatingRequest(b.build(), resultTracker, cameraHandler)
  }

  private val resultTracker = object : CameraCaptureSession.CaptureCallback() {
    override fun onCaptureCompleted(s: CameraCaptureSession, r: CaptureRequest, result: TotalCaptureResult) {
      result.get(CaptureResult.SENSOR_SENSITIVITY)?.let { lastIso = it }
      result.get(CaptureResult.SENSOR_EXPOSURE_TIME)?.let { lastExposureNs = it }
      result.get(CaptureResult.LENS_FOCUS_DISTANCE)?.let { lastFocus = it }
    }
  }

  // ---------------------------------------------------------------- control

  private fun lock(iso: Double?, exposureUs: Double?, focus: Double?, result: MethodChannel.Result) {
    val chars = characteristics ?: return result.error("state", "Camera not initialized", null)
    val isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE) ?: Range(lastIso, lastIso)
    val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
      ?: Range(lastExposureNs, lastExposureNs)
    val framePeriodNs = 1_000_000_000L / fpsRange.upper
    lockIso = (iso?.toInt() ?: lastIso).coerceIn(isoRange.lower, isoRange.upper)
    lockExposureNs = ((exposureUs?.times(1000))?.toLong() ?: lastExposureNs)
      .coerceIn(expRange.lower, min(expRange.upper, framePeriodNs))
    lockFocus = focus?.toFloat() ?: lastFocus
    locked = true
    applyRepeating()
    result.success(
      mapOf(
        "iso" to lockIso.toDouble(),
        "exposureUs" to lockExposureNs / 1000.0,
        "focus" to lockFocus.toDouble(),
        "fps" to fpsRange.upper.toDouble(),
        "manual" to manualSensor,
      ),
    )
  }

  /**
   * Torch pulse on a dedicated urgent-priority looper. The off edge is posted
   * with postAtTime against uptimeMillis; both edges are reported in the
   * sensor timestamp base so they align with frame timestamps.
   */
  private fun flashPulse(durationMs: Int, result: MethodChannel.Result) {
    val handler = torchHandler ?: return result.error("state", "Camera not initialized", null)
    handler.post {
      val onNs = nowNs()
      val onUptime = SystemClock.uptimeMillis()
      torchOn = true
      cameraHandler?.post { applyRepeating() }
      handler.postAtTime({
        val offNs = nowNs()
        torchOn = false
        cameraHandler?.post { applyRepeating() }
        main.post { result.success(mapOf("onUs" to onNs / 1000, "offUs" to offNs / 1000)) }
      }, onUptime + durationMs)
    }
  }

  // ---------------------------------------------------------------- frames

  private fun onImage(r: ImageReader) {
    val image = r.acquireLatestImage() ?: return
    try {
      val out = sink ?: return
      val plane = image.planes[0]
      val rowStride = plane.rowStride
      val pixelStride = plane.pixelStride
      val buf = plane.buffer
      val w = image.width
      val h = image.height
      val s = min(roiSize, min(w, h))
      val x0 = (w - s) / 2
      val y0 = (h - s) / 2
      val roi = ByteArray(s * s)
      // Copy the centre crop and rotate it upright in the same pass so the
      // eye's horizontal meridian is horizontal for the iris-scale search.
      val turns = (sensorOrientation / 90) % 4
      for (y in 0 until s) {
        val rowBase = (y0 + y) * rowStride + x0 * pixelStride
        for (x in 0 until s) {
          val v = buf.get(rowBase + x * pixelStride)
          val (dx, dy) = when (turns) {
            1 -> Pair(s - 1 - y, x)
            2 -> Pair(s - 1 - x, s - 1 - y)
            3 -> Pair(y, s - 1 - x)
            else -> Pair(x, y)
          }
          roi[dy * s + dx] = v
        }
      }
      frameIndex += 1
      val event = mapOf(
        "w" to s,
        "h" to s,
        "ts" to image.timestamp / 1000,
        "idx" to frameIndex,
        "torch" to torchOn,
        "bytes" to roi,
      )
      main.post { sink?.success(event) }
    } finally {
      image.close()
    }
  }

  override fun onListen(arguments: Any?, events: EventChannel.EventSink?) { sink = events }

  override fun onCancel(arguments: Any?) { sink = null }

  // ---------------------------------------------------------------- utils

  /** Now, in the same clock base the sensor stamps frames with. */
  private fun nowNs(): Long =
    if (realtimeTimestamps) SystemClock.elapsedRealtimeNanos() else System.nanoTime()

  private fun release() {
    torchOn = false
    locked = false
    try { session?.close() } catch (_: Exception) {}
    session = null
    device?.close(); device = null
    reader?.close(); reader = null
    previewSurface?.release(); previewSurface = null
    textureEntry?.release(); textureEntry = null
    cameraThread?.quitSafely(); cameraThread = null; cameraHandler = null
    torchThread?.quitSafely(); torchThread = null; torchHandler = null
  }
}
