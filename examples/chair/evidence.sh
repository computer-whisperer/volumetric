#!/bin/bash
# Build the chair's evidence project from the chairbase-dslr-1 session:
# survey the stills here, then import the trained splat (as surfels) and
# its TSDF surface cloud from the cluster run. Writes chair_evidence.vproj
# (about 120 MB with the splat) into the session's demo directory, or the
# path given as the first argument. Needs the scanner tree at $SCAN
# (default /ceph/christian/index_scanner).
set -e
L="$(cd "$(dirname "$0")" && pwd)"
cd "$L/../.."
V=./target/release/volumetric_cli
SCAN=${SCAN:-/ceph/christian/index_scanner}
SESSION=$SCAN/sessions/chairbase-dslr-1
RUN=$SCAN/runs/chairbase-dslr-1-2dgs
P=${1:-$SESSION/demo/chair_evidence.vproj}
mkdir -p "$(dirname "$P")"
rm -f "$P"

$V project-new --output "$P" >/dev/null
# 44 stills, one camera (the JPEGs carry no EXIF; the focal is solved).
$V view-import --stills "$SESSION" --embed preview -p "$P"
$V view-detect -p "$P" | tail -1
$V view-survey -p "$P" --report "${P%.vproj}_survey.json" | grep -E '^(survey done|camera|card):'

# The splat: gsplat's 2DGS export carries an untrained third scale, so the
# kind must be said. The view set gives the world frame and provenance.
$V project-add-asset -p "$P" -i "$RUN/splat.ply" --type blob --asset-id splat_ply >/dev/null
$V project-add-op -p "$P" --operator splat_import_operator -i asset:splat_ply -i asset:views \
    -i 'json:{"kind":"surfel","session":"chairbase-dslr-1","training":"gsplat 2dgs 15000 steps, subject mask, run chairbase-dslr-1-2dgs"}' \
    --output-id splat >/dev/null
# The trainer's surface cloud (TSDF over its own renders, 1 mm) and the
# splat's own opaque centres.
$V project-add-asset -p "$P" -i "$RUN/cloud.ply" --type blob --asset-id cloud_ply >/dev/null
$V project-add-op -p "$P" --operator point_cloud_import_operator -i asset:cloud_ply -i none --output-id tsdf_cloud >/dev/null
$V project-add-op -p "$P" --operator splat_points_operator -i asset:splat -i asset:views -i 'json:{"min_opacity":0.5}' --output-id splat_points >/dev/null

$V project-run -p "$P" | tail -4
$V splat-list -i "$P" | head -3

# The full shoot (192 stills, the 44 above among them) surveyed as a
# standalone set, and ten of its views, previews embedded, for the base
# project (base.sh loads them): the top-down, three low elevations and
# six around the high ring.
ALL=$SCAN/sessions/chairbase-dslr-1-all
A=$ALL/demo/chair_survey_all.vviews
mkdir -p "$ALL/demo"
$V view-import --stills "$ALL" --embed none -o "$A" | tail -1
$V view-detect -i "$A" -o "$A" | tail -1
$V view-survey -i "$A" -o "$A" --report "${A%.vviews}_survey.json" | grep -E '^(survey done|camera):'
$V view-select -i "$A" --posed --embed preview -o "$ALL/demo/chair_views.vviews" \
    --id DSC00742 --id DSC00730 --id DSC00754 --id DSC00723 --id DSC00739 \
    --id DSC00749 --id DSC00727 --id DSC00745 --id DSC00760 --id DSC00731
