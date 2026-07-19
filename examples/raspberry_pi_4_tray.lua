-- Parametric, open-top Raspberry Pi 4 Model B tray.
--
-- All dimensions are metres. The board envelope, 3 mm corner radius, and
-- 58 x 49 mm mounting-hole pattern follow Raspberry Pi's official mechanical
-- drawing. Connector apertures include printable clearance and are intentionally
-- generous: verify them against the exact board revision and printer before use.
--
-- Official reference:
-- https://pip-assets.raspberrypi.com/categories/545-raspberry-pi-4-model-b/documents/RP-008343-DS-1-raspberry-pi-4-mechanical-drawing.pdf

local board_length = 0.085 -- @param key="rpi4.board_length" min=0.08 max=0.09
local board_width = 0.056 -- @param key="rpi4.board_width" min=0.05 max=0.06
local board_clearance = 0.0005 -- @param key="case.board_clearance" min=0.0001 max=0.002
local wall_thickness = 0.002 -- @param key="case.wall_thickness" min=0.0012 max=0.005
local floor_thickness = 0.002 -- @param key="case.floor_thickness" min=0.0012 max=0.005
local case_height = 0.023 -- @param key="case.height" min=0.012 max=0.04
local case_corner_radius = 0.005 -- @param key="case.corner_radius" min=0.003 max=0.01

local standoff_height = 0.003 -- @param key="case.standoff_height" min=0.001 max=0.008
local post_outer_diameter = 0.006 -- @param key="case.post_outer_diameter" min=0.004 max=0.01
local screw_hole_diameter = 0.0029 -- @param key="case.screw_hole_diameter" min=0.0024 max=0.004

local port_clearance = 0.0006 -- @param key="case.port_clearance" min=0.0002 max=0.002
local vent_width = 0.0025 -- @param key="case.vent_width" min=0.0012 max=0.005
local vent_length = 0.036 -- @param key="case.vent_length" min=0.015 max=0.055
local vent_pitch = 0.008 -- @param key="case.vent_pitch" min=0.005 max=0.012

local inner_min_x = -board_clearance
local inner_max_x = board_length + board_clearance
local inner_min_y = -board_clearance
local inner_max_y = board_width + board_clearance
local outer_min_x = inner_min_x - wall_thickness
local outer_max_x = inner_max_x + wall_thickness
local outer_min_y = inner_min_y - wall_thickness
local outer_max_y = inner_max_y + wall_thickness
local inner_corner_radius = math.max(0.0005, case_corner_radius - wall_thickness)

local board_z = floor_thickness + standoff_height
local post_radius = post_outer_diameter / 2.0
local screw_radius = screw_hole_diameter / 2.0

-- Official mounting-hole centers relative to the board's lower-left corner.
local mount_left_x = 0.0035
local mount_right_x = mount_left_x + 0.058
local mount_bottom_y = 0.0035
local mount_top_y = mount_bottom_y + 0.049

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
    return disk(x, y, nearest_x, nearest_y, radius)
end

function rounded_rectangle(x, y, min_x, max_x, min_y, max_y, radius)
    local nearest_x = math.max(min_x + radius, math.min(max_x - radius, x))
    local nearest_y = math.max(min_y + radius, math.min(max_y - radius, y))
    return disk(x, y, nearest_x, nearest_y, radius)
end

function box(x, y, z, min_x, max_x, min_y, max_y, min_z, max_z)
    return x >= min_x and x <= max_x and
        y >= min_y and y <= max_y and
        z >= min_z and z <= max_z
end

function mounting_post(x, y, z)
    local inside_height = z >= floor_thickness and z <= board_z
    return inside_height and (
        disk(x, y, mount_left_x, mount_bottom_y, post_radius) or
        disk(x, y, mount_left_x, mount_top_y, post_radius) or
        disk(x, y, mount_right_x, mount_bottom_y, post_radius) or
        disk(x, y, mount_right_x, mount_top_y, post_radius)
    )
end

function screw_hole(x, y, z)
    local inside_height = z >= 0.0 and z <= board_z
    return inside_height and (
        disk(x, y, mount_left_x, mount_bottom_y, screw_radius) or
        disk(x, y, mount_left_x, mount_top_y, screw_radius) or
        disk(x, y, mount_right_x, mount_bottom_y, screw_radius) or
        disk(x, y, mount_right_x, mount_top_y, screw_radius)
    )
end

function floor_vent(x, y, z)
    local vent_start_x = (board_length - vent_length) / 2.0
    local vent_end_x = vent_start_x + vent_length
    local first_vent_y = 0.012
    local vent = false
    for row = 0, 4 do
        vent = vent or capsule(
            x, y,
            vent_start_x, first_vent_y + row*vent_pitch,
            vent_end_x, first_vent_y + row*vent_pitch,
            vent_width/2.0
        )
    end
    return z >= 0.0 and z <= floor_thickness and vent
end

function connector_cutout(x, y, z)
    local side_min_x = inner_max_x - port_clearance
    local side_max_x = outer_max_x + port_clearance
    local edge_min_y = outer_min_y - port_clearance
    local edge_max_y = inner_min_y + port_clearance

    -- Right edge: lower/upper USB stacks and Ethernet. These retain narrow
    -- wall ribs between apertures instead of removing the whole side.
    local lower_usb = box(
        x, y, z,
        side_min_x, side_max_x,
        0.001 - port_clearance, 0.017 + port_clearance,
        board_z - 0.001, board_z + 0.019
    )
    local upper_usb = box(
        x, y, z,
        side_min_x, side_max_x,
        0.018 - port_clearance, 0.034 + port_clearance,
        board_z - 0.001, board_z + 0.019
    )
    local ethernet = box(
        x, y, z,
        side_min_x, side_max_x,
        0.036 - port_clearance, board_width + port_clearance,
        board_z - 0.001, board_z + 0.017
    )

    -- Lower edge: USB-C power, two micro-HDMI ports, and the A/V jack.
    local usb_c = box(
        x, y, z,
        0.005 - port_clearance, 0.016 + port_clearance,
        edge_min_y, edge_max_y,
        board_z - 0.001, board_z + 0.0045
    )
    local hdmi_zero = box(
        x, y, z,
        0.022 - port_clearance, 0.030 + port_clearance,
        edge_min_y, edge_max_y,
        board_z - 0.001, board_z + 0.006
    )
    local hdmi_one = box(
        x, y, z,
        0.035 - port_clearance, 0.044 + port_clearance,
        edge_min_y, edge_max_y,
        board_z - 0.001, board_z + 0.006
    )
    local av_jack = box(
        x, y, z,
        0.050 - port_clearance, 0.058 + port_clearance,
        edge_min_y, edge_max_y,
        board_z - 0.001, board_z + 0.0095
    )

    return lower_usb or upper_usb or ethernet or usb_c or hdmi_zero or hdmi_one or av_jack
end

function is_inside(x, y, z)
    local outer = rounded_rectangle(
        x, y,
        outer_min_x, outer_max_x,
        outer_min_y, outer_max_y,
        case_corner_radius
    ) and z >= 0.0 and z <= case_height

    local cavity = rounded_rectangle(
        x, y,
        inner_min_x, inner_max_x,
        inner_min_y, inner_max_y,
        inner_corner_radius
    ) and z >= floor_thickness and z <= case_height

    local structure = (outer and not cavity) or mounting_post(x, y, z)
    local opening = screw_hole(x, y, z) or floor_vent(x, y, z) or connector_cutout(x, y, z)
    return structure and not opening
end

function get_bounds_min_x() return outer_min_x end
function get_bounds_max_x() return outer_max_x end
function get_bounds_min_y() return outer_min_y end
function get_bounds_max_y() return outer_max_y end
function get_bounds_min_z() return 0.0 end
function get_bounds_max_z() return case_height end
