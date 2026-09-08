-- Toy car side profile: x along the car (nose at +x), y up. Metres.
local chassis_x0 = -0.040 -- @param key="car.tail_x" min=-0.1 max=0.0
local chassis_y0 = 0.006
local chassis_y1 = 0.023
local nose_cx = 0.0315
local nose_cy = (chassis_y0 + chassis_y1) / 2.0
local nose_r = (chassis_y1 - chassis_y0) / 2.0
local cabin_x0 = -0.028
local cabin_x1 = 0.006
local cabin_y1 = 0.037
local roof_x0 = -0.024
local roof_x1 = -0.002

function is_inside(x, y)
    local chassis = x >= chassis_x0 and x <= nose_cx and y >= chassis_y0 and y <= chassis_y1
    local dx = x - nose_cx
    local dy = y - nose_cy
    local nose = dx*dx + dy*dy <= nose_r*nose_r
    local t = (y - chassis_y1) / (cabin_y1 - chassis_y1)
    local rear = cabin_x0 + t*(roof_x0 - cabin_x0)
    local front = cabin_x1 + t*(roof_x1 - cabin_x1)
    local cabin = y >= chassis_y1 and y <= cabin_y1 and x >= rear and x <= front
    return chassis or nose or cabin
end

function get_bounds_min_x() return chassis_x0 end
function get_bounds_max_x() return nose_cx + nose_r end
function get_bounds_min_y() return chassis_y0 end
function get_bounds_max_y() return cabin_y1 end
