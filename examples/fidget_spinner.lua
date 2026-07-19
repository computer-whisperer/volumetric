-- Parameterized three-lobed fidget spinner with four bearing seats.
-- All dimensions are metres. Changing the constants in this first block
-- updates the solid, its clearances, and its declared bounds together.

local bearing_outer_diameter = 0.022 -- @param key="spinner.bearing_outer_diameter" min=0.005 max=0.05
local radial_clearance = 0.00015 -- @param key="spinner.radial_clearance" min=0.0 max=0.002
local bearing_pitch = 0.035 -- @param key="spinner.bearing_pitch" min=0.02 max=0.1
local body_thickness = 0.0072 -- @param key="spinner.body_thickness" min=0.001 max=0.03

local center_outer_radius = 0.0165 -- @param key="spinner.center_outer_radius" min=0.005 max=0.05
local lobe_outer_radius = 0.0155 -- @param key="spinner.lobe_outer_radius" min=0.005 max=0.05
local web_radius = 0.0105 -- @param key="spinner.web_radius" min=0.002 max=0.03
local bounds_margin = 0.001

local bearing_radius = bearing_outer_diameter / 2.0 + radial_clearance
local half_thickness = body_thickness / 2.0
local half_pitch = bearing_pitch / 2.0
local lobe_y = bearing_pitch * math.sqrt(3.0) / 2.0
local bounds_x = bearing_pitch + lobe_outer_radius + bounds_margin
local bounds_y = lobe_y + lobe_outer_radius + bounds_margin

function disk(x, y, center_x, center_y, radius)
    local dx = x - center_x
    local dy = y - center_y
    return dx*dx + dy*dy <= radius*radius
end

function capsule(x, y, start_x, start_y, end_x, end_y, radius)
    local segment_x = end_x - start_x
    local segment_y = end_y - start_y
    local from_start_x = x - start_x
    local from_start_y = y - start_y
    local length_squared = segment_x*segment_x + segment_y*segment_y
    local along = (from_start_x*segment_x + from_start_y*segment_y) / length_squared
    along = math.max(0.0, math.min(1.0, along))

    local nearest_x = start_x + along*segment_x
    local nearest_y = start_y + along*segment_y
    local dx = x - nearest_x
    local dy = y - nearest_y
    return dx*dx + dy*dy <= radius*radius
end

function is_inside(x, y, z)
    local right_x = bearing_pitch
    local right_y = 0.0
    local upper_left_x = -half_pitch
    local upper_left_y = lobe_y
    local lower_left_x = -half_pitch
    local lower_left_y = -lobe_y

    local body =
        disk(x, y, 0.0, 0.0, center_outer_radius) or
        disk(x, y, right_x, right_y, lobe_outer_radius) or
        disk(x, y, upper_left_x, upper_left_y, lobe_outer_radius) or
        disk(x, y, lower_left_x, lower_left_y, lobe_outer_radius) or
        capsule(x, y, 0.0, 0.0, right_x, right_y, web_radius) or
        capsule(x, y, 0.0, 0.0, upper_left_x, upper_left_y, web_radius) or
        capsule(x, y, 0.0, 0.0, lower_left_x, lower_left_y, web_radius)

    local bearing_hole =
        disk(x, y, 0.0, 0.0, bearing_radius) or
        disk(x, y, right_x, right_y, bearing_radius) or
        disk(x, y, upper_left_x, upper_left_y, bearing_radius) or
        disk(x, y, lower_left_x, lower_left_y, bearing_radius)

    return math.abs(z) <= half_thickness and body and not bearing_hole
end

function get_bounds_min_x() return -bounds_x end
function get_bounds_max_x() return bounds_x end
function get_bounds_min_y() return -bounds_y end
function get_bounds_max_y() return bounds_y end
function get_bounds_min_z() return -half_thickness end
function get_bounds_max_z() return half_thickness end
