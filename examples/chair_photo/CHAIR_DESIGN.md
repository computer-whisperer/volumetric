# Custom chair — engineering starting point

This is a discussion brief, not a ratified chair design or fabrication release.
The base is now an articulated reference assembly. The next design is an upper
chair that supports the user's existing wheelchair cushions. Cushion geometry,
loaded thickness, intended users/load, posture goals, fabrication process and
required adjustment ranges are still awaiting user input.

## Interfaces we can design against

All dimensions below are observations of the reference base, not specifications
for a purchased mechanism's strength or manufacturing tolerances. Millimetres;
seat-local X is across the front crossbar, Y points toward the receiver.

| Interface | Existing evidence | Consequence for the next model |
|---|---|---|
| Front outer slots | Centers about 213.6 apart; Z about +3.5/+3.7 relative to inner-hole datum | Define adapter contact pads/spacers; do not force a flat plate onto all six stations |
| Front inner holes | Centers about 109.8 apart; local Z=0 defines the datum | Candidate alternate attachment pattern; confirm which stations are intended structural mounts |
| Rear rail slots | Centers about 29.9 apart at Y about 174.6 | Rear attachment is narrow; carrier must distribute cushion loads into the actual support footprint |
| Receiver mouth | Fitted outer entry rim about 39 x 18, rounded profile | Entry rim is not throat clearance; do not dimension the final mating tongue from this alone |
| Receiver internals | Throat, insertion depth/direction, clamp contact and positive retention unresolved | The first fit-critical fabrication should be a cheap interface coupon after inspecting the socket |
| Floor-to-seat datum | About 460 at the photographed pose, before cushion/adapter/pan | Establish required loaded seat height and actual gas-lift travel before choosing the vertical stack |

`work/base-interfaces.json` associates the seat datum with `crossbar`, the mouth
with `backrest_receiver`, and the pin reference with `rear_pins`. It records rest
and posed frames. The corresponding metadata is stored in the project. Static
Subspace timeline outputs remain survey datums; they do not automatically move
with the assembly. A new part and its mating datum must follow the same owner.

The lift axis intersects the seat datum at approximately X=0.0, Y=52.8 mm.
The attachment polygon is not a cushion-centering prescription: cushion
placement needs the intended occupied-seat reference and resulting load case.
`plot_interfaces.py` (optional Matplotlib dependency) regenerates the PNG/SVG
interface map in work/ from the measurement reports.

## Proposed decomposition to discuss

1. **Cushion carrier:** its support surface and retention features follow the
   actual cushion's required underside support and removable cover/interface.
   Decide these from the existing cushion product, not from a generic cushion
   rectangle. Locate its occupied-seat reference relative to the lift axis.
2. **Base adapter:** a replaceable connection between carrier and the confirmed
   mounting stations. Set the contact pads and vertical offset explicitly;
   choose fasteners only after checking hole type, thread engagement/access and
   the purchased mechanism's intended attachment pattern. Keep the mechanism's
   controls reachable throughout the required adjustment envelope.
3. **Backrest spine and support:** use the purchased tilt mechanism only after
   confirming which components move, their pivots/range and the receiver's real
   interface. Keep the socket mating piece replaceable while that fit is being
   established. The backrest load path and any restraint/retention must be
   deliberate; a friction clamp or guessed insertion depth is not verified here.

The expected seat load path is cushion → carrier → adapter/mounting bolts →
purchased mechanism → lift → star → casters. The backrest adds a separate
moment through its spine and receiver. This decomposition lets us revise one
interface without remaking the cushion carrier.

## Decisions that determine geometry

- Which cushion(s): outline, underside support shape, loaded thickness, cover
  overhang and required attachment/retention. Does one chair accept multiple
  sizes/products, or one selected cushion?
- Intended user/load range and use: transfers, leaning, recline, side loading,
  adjustable arm supports, foot support, or other specialty requirements.
- Required loaded seat-height range, seat depth/fore-aft position, seat angle,
  backrest angle/height, and their adjustment versus fixed settings.
- Available processes and materials for structural pieces, prototype process,
  and whether a replaceable metal adapter/tongue is acceptable.

The key stack is the support height at a joint state plus adapter/pan offset
plus the **loaded** cushion thickness. The photographed 460 mm datum is not the
mechanism's minimum height. The assembly's -20..+80 mm exploration range is not
a measured gas-lift specification. A tall cushion can change whether this base
is suitable even if the bolt pattern fits.

## First useful downstream modelling pass

After those inputs, model the actual cushion support envelope and a simple
carrier/adapter envelope; place them relative to the measured holes and lift
axis. Set named cushion-seat and backrest datums owned by their moving parts.
Use the joint tree to check reach and interference in candidate settings. Keep
backrest tilt fixed until the physical linkage is characterized; do not invent
synchro-tilt ratios from these still photographs.

Before choosing structural section sizes, establish load cases and check the
carrier's bending, bolt-group loads and local pull-through, receiver/spine
bending and retention, and whole-chair stability over caster orientations and
recline. Current kinematics enforce neither ground contact nor stability, and
rigid motion is not a strength calculation. No material thickness, rated load,
or safety margin has been selected in this brief.

Favor evidence that repeats: cushion drawings/CAD already in use, photographs
showing each mechanism extreme and underside, and a reusable interface coupon.
Target any manual measurement at uncertainty that blocks a mating part rather
than trying to measure the entire base again.
