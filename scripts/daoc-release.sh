#!/bin/bash
# Cut a DAoC build release of this fork: tag, build the bundle, write
# release.json and release notes, and print the push and `gh` commands.
#
# Usage: scripts/daoc-release.sh vX.Y.Z-daoc.N
#
# Run from a clean tree on the commit to release. The tag is created locally
# (annotated) and deleted again if anything after that fails. Nothing is pushed
# and `gh` is never run: the commands to publish are printed at the end.
#
# WINE_X86_64 names the x86_64 Wine whose `winebuild` marks the bundle's
# builtins (default: the daoc checkout's dist/wine). Only winebuild is run.
set -euo pipefail

cd "$(dirname "$0")/.."

die() { echo "daoc-release: $*" >&2; exit 1; }

[ $# -eq 1 ] || die "usage: $0 vX.Y.Z-daoc.N"
TAG=$1

[[ $TAG =~ ^v([0-9]+\.[0-9]+\.[0-9]+)-daoc\.[0-9]+$ ]] || die "tag '$TAG' does not match vX.Y.Z-daoc.N"
BASE=${BASH_REMATCH[1]}
for ws in windows unix; do
    v=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ws/Cargo.toml" | head -1)
    [ "$v" = "$BASE" ] || die "$ws/Cargo.toml says $v, the tag's base is $BASE"
done

[ -z "$(git status --porcelain)" ] || die "the tree is dirty; commit or stash first"
! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || die "tag $TAG already exists"

WINE_X86_64=${WINE_X86_64:-$PWD/../../../dist/wine}
WINE_X86_64=$(cd "$WINE_X86_64" 2>/dev/null && pwd) || die "WINE_X86_64 does not exist"
[ -x "$WINE_X86_64/bin/winebuild" ] || die "$WINE_X86_64/bin/winebuild is missing"

git tag -a "$TAG" -m "DAoC build $TAG"
created=1
cleanup() {
    if [ "${created:-0}" = 1 ]; then
        git tag -d "$TAG" >/dev/null && echo "daoc-release: failed; deleted tag $TAG" >&2
    fi
}
trap cleanup EXIT

[ "$(git describe --tags --always)" = "$TAG" ] || die "git describe does not name $TAG"

# unix/shared/build.rs watches refs/tags, so the new tag reruns it and every
# binary is restamped; the bundle target refuses any binary not stamped $TAG.
PATH=$HOME/.cargo/bin:$PATH make PROD=1 WINE_X86_64="$WINE_X86_64" \
    RUST_STABLE=1.99.0 RUST_NIGHTLY=nightly-2026-10-11 CARGO_TOOLS=xwin@0.10.0 bundle

OUT=windows/target
ASSET=mtld3d.tar.xz
[ -f "$OUT/$ASSET" ] || die "$OUT/$ASSET was not produced"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"; cleanup' EXIT
tar -xJf "$OUT/$ASSET" -C "$STAGE"

/usr/bin/python3 - "$TAG" "$ASSET" "$OUT/$ASSET" "$STAGE" "$OUT/release.json" <<'PY'
import hashlib, json, os, sys
tag, asset, archive, stage, out = sys.argv[1:]
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
members = {}
for root, _, files in os.walk(stage):
    for name in files:
        path = os.path.join(root, name)
        members[os.path.relpath(path, stage)] = sha(path)
doc = {"version": tag, "asset": asset, "sha256": sha(archive),
       "members": dict(sorted(members.items()))}
with open(out, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
PY

NOTES=$OUT/release-notes.md
cat > "$NOTES" <<EOF
mtld3d $TAG: a modified build of [mtld3d](https://github.com/athei/mtld3d) $BASE for Dark Age of Camelot.

Adds per-pixel fixed-function lighting and DAoC light controls, driven by \`lighting.*\` keys in \`mtld3d.conf\`, all off by default. See "This build" in README.md.

Built from $(git rev-parse --short HEAD). \`release.json\` lists the sha256 of the archive and of every file in it.
EOF

created=0
echo
echo "daoc-release: $TAG built; $OUT/$ASSET, $OUT/release.json, $NOTES"
echo "Publish with:"
echo "  git push personal ppl $TAG"
echo "  gh release create $TAG --repo trozzelle/mtld3d --title $TAG --notes-file $NOTES $OUT/$ASSET $OUT/release.json"
