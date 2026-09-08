-- Side window cutouts in the car's side view: x along the car, y up. Metres.
-- Rear window is a rectangle; the front window's leading edge is parallel
-- to the windscreen rake ((6,23)->(-2,37): dx/dy = -8/14).
local sill_y = 0.025
local top_y = 0.035
local rear_x0 = -0.025
local rear_x1 = -0.012
local front_x0 = -0.009
local front_x1_at_sill = 0.003
local rake = -8.0 / 14.0
local sill_from_deck = sill_y - 0.023

function is_inside(x, y)
    local band = y >= sill_y and y <= top_y
    local rear = x >= rear_x0 and x <= rear_x1
    local lead = front_x1_at_sill + (y - sill_y) * rake
    local front = x >= front_x0 and x <= lead
    return band and (rear or front)
end

function get_bounds_min_x() return rear_x0 end
function get_bounds_max_x() return front_x1_at_sill end
function get_bounds_min_y() return sill_y end
function get_bounds_max_y() return top_y end
