-- Wheel cross-section for revolve: x = radius from the axle, y = position
-- along the axle (outer face at +y). Metres.
local tyre_r = 0.009
local half_w = 0.004
local shoulder_r = 0.0015
local groove_a = 0.0014
local groove_half = 0.0006
local groove_depth = 0.001
local dish_r = 0.0055
local dish_floor = 0.0028
local boss_r = 0.0018

function is_inside(r, a)
    local aa = math.abs(a)
    if r > tyre_r or aa > half_w then
        return 0.0
    end
    -- Rounded tyre shoulders.
    local sx = math.max(r - (tyre_r - shoulder_r), 0.0)
    local sy = math.max(aa - (half_w - shoulder_r), 0.0)
    if sx*sx + sy*sy > shoulder_r*shoulder_r then
        return 0.0
    end
    -- Two circumferential tread grooves.
    if math.abs(aa - groove_a) <= groove_half and r > tyre_r - groove_depth then
        return 0.0
    end
    -- Hub dish on the outer face, with a centre boss left standing.
    if a > dish_floor and r < dish_r and r > boss_r then
        return 0.0
    end
    return 1.0
end

function get_bounds_min_x() return 0.0 end
function get_bounds_max_x() return tyre_r end
function get_bounds_min_y() return -half_w end
function get_bounds_max_y() return half_w end
