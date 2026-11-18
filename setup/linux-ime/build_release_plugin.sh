#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_root=${1:?pass an empty temporary build directory}
mkdir -p -- "$build_root/sdk" "$build_root/fcitx"
cd -- "$build_root"

qt_package=qtbase-devel-6.9.2-hfd094ee_5.conda
fcitx_revision=c328b03f32a4276d9aeb3213d1f3f959ef6848f2
curl -fsSL --retry 3 --max-time 180 "https://repo.anaconda.com/pkgs/main/linux-64/$qt_package" -o "$qt_package"
curl -fsSL --retry 3 --max-time 180 "https://codeload.github.com/fcitx/fcitx-qt5/tar.gz/$fcitx_revision" -o fcitx.tar.gz
sha256sum --check <<'CHECKSUMS'
7fcf42b85f60f89bd85f418a57a47189ebb0ddb12ea7f8ed4ac3332cd6ccd104  qtbase-devel-6.9.2-hfd094ee_5.conda
2dd423baccd51584932769ffa9eefae03a1b83295dfd7a09c04db9519be23281  fcitx.tar.gz
CHECKSUMS
unzip -p "$qt_package" 'pkg-*.tar.zst' | tar --zstd -xf - -C sdk
tar -xzf fcitx.tar.gz --strip-components=1 -C fcitx
python "$repo_root/setup/linux-ime/build_fcitx.py" --source "$PWD/fcitx" --qt-sdk "$PWD/sdk"
