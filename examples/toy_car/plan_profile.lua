-- Toy car plan outline: x along the car, y here is world z (width). Metres.
local half_length = 0.040
local half_width = 0.016
local corner_r = 0.006

function is_inside(x, y)
    local dx = math.max(math.abs(x) - (half_length - corner_r), 0.0)
    local dy = math.max(math.abs(y) - (half_width - corner_r), 0.0)
    return dx*dx + dy*dy <= corner_r*corner_r
end

function get_bounds_min_x() return -half_length end
function get_bounds_max_x() return half_length end
function get_bounds_min_y() return -half_width end
function get_bounds_max_y() return half_width end
