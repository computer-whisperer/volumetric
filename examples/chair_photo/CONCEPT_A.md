# Chair concept A — first layout, 2026-09-12

Authorized scope: a WGSL assembly extending the photo-derived base, accepting
14–20 inch cushions and exploring a 22 inch carrier. This is an iteration model,
not a fabrication release or a rated chair. Preserve the existing base and its
measured holes. Keep the purchased backrest tilt fixed until its motion is known.

## Chosen architecture

One continuous folded metal tray, two longitudinal metal rails, and transverse
mount adapters distribute load to the measured outer front and rear mounting
stations. A 20 inch cushion has 4 mm edge allowance on every side. Smaller
cushions can use this same tray; 14 and 22 inch tray variants trade bulk against
support area. Rear tray edge stays at seat-local Y=235 mm, ahead of the receiver
at Y=275 mm. The front overhang changes with depth; do not center a large tray on
the lift axis and inadvertently put the backrest mast through the cushion.

Initial sections are layout choices: 3 mm tray, 20 mm downturned side/rear lips,
20 x 30 x 2 mm longitudinal tubes, 6 mm mount crossmembers. Support top is 43 mm
above the measured inner-hole datum. Four pan screws have conical head envelopes
0.3 mm below that plane; their countersinks continue into backing pads attached
to the rails. Fastener heads, threads, welds, bends and local bearing need detail
design. Countersinking unsupported thin sheet is not the assumed construction.

Arm supports attach to the chassis, outside the pan. Each telescoping post has
80 mm illustrative height travel. Pad carriages move inward over the pan, so a
14 inch cushion can have closer arm spacing without moving a post through the
seat. Left/right pad inset is coupled in CAD for symmetric setup; this does not
imply a mechanical linkage. Pads and guide housings are compact printed parts;
the posts and chassis are metal. Locking hardware is an unresolved detail.

Keep the existing printed backrest, four-lobe slotted plate, clamp and transverse
tube. Photos suggest a roughly 200 x 76 mm fastener rectangle, 251 mm long tube
and 22 mm apparent diameter. Additional-view bolt residuals are much worse than
the card fit: these dimensions are approximate. A two-collar printed yoke grips
the tube near its ends, supported by a metal mast behind the retained clamp. A sloping lower connector
returns to the receiver shoe, keeping the mast clear of the existing plate. The yoke/spine assembly has
80 mm illustrative backrest height adjustment. Both the arm and backrest
telescoping sections retain at least 50 mm engagement over that travel; this
is a geometric constraint, not a verified strength criterion. Its tongue is a replaceable
metal part aligned with the measured receiver mouth; 32 x 10 mm section and
45 mm insertion are assumptions, pending a fit coupon and retention design.

The shell in CAD is a clearance proxy informed by the 15 x 15 inch label and
visible wings, using the labelled 6 inch lateral depth for the forward wing
envelope. Arm supports sit 50 mm farther forward to clear that envelope. It is not a reconstruction or replacement manufacturing file.
The cushion is a separate 80 mm thick envelope, hidden in the engineering view.

## Open engineering questions

- Occupied seat reference and stability: the 20 inch tray center is about 76 mm
  forward of the lift axis. Larger trays increase the forward overhang. No
  occupied center of mass, caster support polygon margin or tipping test yet.
- Height: photographed datum 460 + carrier 43 + assumed cushion 80 = roughly
  583 mm. Actual minimum lift height and loaded cushion thickness remain unknown.
- Loads/materials: no rated load or verified section sizes. Check rail bending,
  adapter fasteners, arm transfer loads, yoke fatigue, receiver moment and
  positive retention before use. Printed structural parts need material and
  orientation selection, coupons and appropriate validation.
- Confirm intended base mounting stations, threads and access; hidden socket
  dimensions and clamp engagement are still the most consequential fit unknowns.
- Carbon bounding box limit is 410 x 256 x 460 mm. Small FDM bed size is unknown;
  fit coupons are a better first print than a complete load-bearing component.

## Evidence and replay

`measure_backrest.py` recognizes the calibrated physical card through the
inventory in `../calibration-targets.json`, records native ViewSet picks and
checks, and generates `backrest-report.json`. All pixels refer to the encoded
6192 x 4128 JPEG, not an EXIF-rotated display. See the README for replay commands.
`build_concept.py` produces the downstream project, flattened articulated
assembly, dimensions and print envelopes. `audit_concept.py` checks support-plane
coverage, recessed heads, attachment geometry and independent joint motion.
