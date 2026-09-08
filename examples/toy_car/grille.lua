-- Grille slots on the nose, sketched in the car's front view: x here is
-- world z (width), y is world y (height). Three 0.8 mm slots between the
-- headlights and the bumper line. Metres.
local half_span = 0.007
local slot_h = 0.0008
local slot0_y = 0.0100
local pitch = 0.0015
local margin = 0.0005

function is_inside(x, y)
    if math.abs(x) > half_span then
        return 0.0
    end
    local k = math.floor((y - slot0_y) / pitch + 0.5)
    if k < 0 or k > 2 then
        return 0.0
    end
    local centre = slot0_y + k * pitch
    return math.abs(y - centre) <= slot_h / 2.0
end

function get_bounds_min_x() return -half_span - margin end
function get_bounds_max_x() return half_span + margin end
function get_bounds_min_y() return slot0_y - pitch end
function get_bounds_max_y() return slot0_y + 3.0 * pitch end
