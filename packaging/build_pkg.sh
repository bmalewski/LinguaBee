#!/usr/bin/env bash
# Buduje instalator macOS: dist/LinguaBee-<wersja>.pkg z gotowego dist/LinguaBee.app.
#
# Użycie:
#   packaging/build_pkg.sh              # wymaga wcześniejszego packaging/build_macos.sh
#
# Instalator (niepodpisany) kopiuje aplikację do /Applications. Używa wyłącznie narzędzi
# systemowych macOS (pkgbuild, productbuild). Wersja jest odczytywana z Info.plist aplikacji.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APP="$ROOT/dist/LinguaBee.app"
if [ ! -d "$APP" ]; then
  echo "Brak $APP — najpierw uruchom packaging/build_macos.sh" >&2
  exit 1
fi

IDENTIFIER="pl.malewski.linguabee"
VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP/Contents/Info.plist")"
OUT="$ROOT/dist/LinguaBee-$VERSION.pkg"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/linguabee-pkg.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

echo "== pkg: LinguaBee $VERSION =="

# 1) Katalog źródłowy z samą aplikacją (payload instalowany do /Applications).
STAGE="$WORK/root"
mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
# Bez rozszerzonych atrybutów (provenance/quarantine): inaczej trafiają do payloadu jako pliki
# AppleDouble (._*) i są odtwarzane na komputerze docelowym.
xattr -cr "$STAGE/LinguaBee.app"

# 2) Plist komponentu: BundleIsRelocatable=false, aby instalator zawsze pisał do
#    /Applications/LinguaBee.app, a nie "aktualizował" innej kopii aplikacji znalezionej
#    gdzie indziej na dysku (np. w katalogu dist/ dewelopera).
COMPONENT_PLIST="$WORK/component.plist"
pkgbuild --analyze --root "$STAGE" "$COMPONENT_PLIST" >/dev/null
/usr/libexec/PlistBuddy -c "Set :0:BundleIsRelocatable false" "$COMPONENT_PLIST"
/usr/libexec/PlistBuddy -c "Set :0:BundleIsVersionChecked false" "$COMPONENT_PLIST"

# 3) Pakiet komponentu.
COMPONENT_PKG="$WORK/LinguaBee-component.pkg"
pkgbuild \
  --root "$STAGE" \
  --component-plist "$COMPONENT_PLIST" \
  --identifier "$IDENTIFIER" \
  --version "$VERSION" \
  --install-location /Applications \
  "$COMPONENT_PKG" >/dev/null

# 4) Dystrybucja: tytuł, ekrany powitalny/końcowy, wymóg Apple Silicon i macOS 14+.
DIST_XML="$WORK/Distribution.xml"
cat > "$DIST_XML" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>LinguaBee $VERSION</title>
  <options customize="never" require-scripts="false" hostArchitectures="arm64" rootVolumeOnly="true"/>
  <domains enable_localSystem="true"/>
  <welcome file="welcome.html" mime-type="text/html"/>
  <conclusion file="conclusion.html" mime-type="text/html"/>
  <volume-check>
    <allowed-os-versions><os-version min="14.0"/></allowed-os-versions>
  </volume-check>
  <choices-outline>
    <line choice="default"><line choice="$IDENTIFIER"/></line>
  </choices-outline>
  <choice id="default"/>
  <choice id="$IDENTIFIER" visible="false"><pkg-ref id="$IDENTIFIER"/></choice>
  <pkg-ref id="$IDENTIFIER" version="$VERSION" onConclusion="none">LinguaBee-component.pkg</pkg-ref>
</installer-gui-script>
EOF

rm -f "$OUT"
productbuild \
  --distribution "$DIST_XML" \
  --resources "$ROOT/packaging/pkg" \
  --package-path "$WORK" \
  "$OUT" >/dev/null

echo "== gotowe =="
du -sh "$OUT"
echo "Instalator: $OUT"
echo "Instalator nie jest podpisany certyfikatem Apple: na innym Macu odblokuj go w"
echo "Ustawieniach systemowych → Prywatność i ochrona → Otwórz mimo to (patrz README, sekcja 10)."
