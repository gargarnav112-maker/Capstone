# Display text is EXACTLY the user's spelling. GURMUKHI is a per-word phonetic
# transliteration used only to drive the acoustic forced aligner.
LINES = [
    "Bebe kehndi tainu vyahuna te mera shastar vi na thaka",
    "Dass ki kar laina kavan ni mera baazan aala raakha",
    "Rahan vich rode behna je yaaran nal paina kass ke",
    "Jatt mudd ton zehri naag ni reha koi sanu dass ke",
    "Baabe di sanu baksh ae kann nattiyan gaani lashke",
    "Ni modhe paundi jhummar dunaali keh ke ashke",
]
GURMUKHI = [
    "ਬੇਬੇ ਕਹਿੰਦੀ ਤੈਨੂੰ ਵਿਆਹੁਣਾ ਤੇ ਮੇਰਾ ਸ਼ਸਤਰ ਵੀ ਨਾ ਥੱਕਾ",
    "ਦੱਸ ਕੀ ਕਰ ਲੈਣਾ ਕਵਣ ਨੀ ਮੇਰਾ ਬਾਜ਼ਾਂ ਆਲਾ ਰਾਖਾ",
    "ਰਾਹਾਂ ਵਿੱਚ ਰੋੜੇ ਬਹਿਣਾ ਜੇ ਯਾਰਾਂ ਨਾਲ ਪੈਣਾ ਕੱਸ ਕੇ",
    "ਜੱਟ ਮੁੱਢ ਤੋਂ ਜ਼ਹਿਰੀ ਨਾਗ ਨੀ ਰਿਹਾ ਕੋਈ ਸਾਨੂੰ ਡੱਸ ਕੇ",
    "ਬਾਬੇ ਦੀ ਸਾਨੂੰ ਬਖ਼ਸ਼ ਏ ਕੰਨ ਨੱਤੀਆਂ ਗਾਨੀ ਲਿਸ਼ਕੇ",
    "ਨੀ ਮੋਢੇ ਪਾਉਂਦੀ ਝੁੰਮਰ ਦੁਨਾਲੀ ਕਹਿ ਕੇ ਅਸ਼ਕੇ",
]
for a, b in zip(LINES, GURMUKHI):
    assert len(a.split()) == len(b.split()), (a, b)
