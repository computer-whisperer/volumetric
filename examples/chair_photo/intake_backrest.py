#!/usr/bin/env python3
"""Extract this session's native camera JPEGs, preserving encoded pixel coordinates.

Requires rawpy (used only for lossless extraction of the embedded camera JPEG).
Camera solving and measurements are performed separately by volumetric.
"""
import argparse
import hashlib
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent


def jpeg_codestream(data):
    """Hashable JPEG coding bytes, excluding APP/COM metadata before the scan.

    LibRaw synthesizes EXIF containing bytes that vary between extractions of
    this ARW. Preserve the JPEG on disk, but verify its deterministic compressed
    image data instead of that metadata. No decoding or recompression occurs.
    """
    if data[:2] != b'\xff\xd8':
        raise ValueError('Not a JPEG')
    result = bytearray(data[:2]); pos = 2
    while pos < len(data):
        start = pos
        if data[pos] != 0xff:
            raise ValueError('Invalid JPEG marker')
        while data[pos] == 0xff:
            pos += 1
        marker = data[pos]; pos += 1
        if marker == 0xda:  # Start of scan: retain entropy stream verbatim.
            return bytes(result)+data[start:]
        length = int.from_bytes(data[pos:pos+2], 'big')
        if length < 2 or pos+length > len(data):
            raise ValueError('Invalid JPEG segment length')
        pos += length
        if not (0xe0 <= marker <= 0xef or marker == 0xfe):
            result.extend(data[start:pos])
    raise ValueError('JPEG has no scan')


def main():
    import rawpy
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('/ceph/christian/Photos/library/2026/2026-09-12'))
    ap.add_argument('--work',type=Path,default=HERE/'work'/'backrest')
    args=ap.parse_args(); dest=args.work/'photos';dest.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HERE/'backrest-photo-manifest.json').read_text())
    for item in manifest['photos']:
        source=args.source/Path(item['source']).name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f'Source hash mismatch: {source}')
        with rawpy.imread(str(source)) as raw:
            thumb=raw.extract_thumb()
        if thumb.format!=rawpy.ThumbFormat.JPEG:
            raise ValueError(f'Expected embedded JPEG: {source}')
        if hashlib.sha256(jpeg_codestream(thumb.data)).hexdigest()!=item['jpeg_codestream_sha256']:
            raise ValueError(f'Embedded JPEG hash mismatch: {source}')
        (dest/item['jpeg']).write_bytes(thumb.data)
    print(f'Verified and extracted {len(manifest["photos"])} camera JPEGs into {dest}')


if __name__=='__main__':main()
