-- Family D outer shell, fully analytic: rounded outer (2mm offset of an
-- inset core box, flat bottom), open-top JST notches on E/S/W, stadium
-- USB opening hugging the connector, flush-top scoop above it.

local wall_r = 0.002 -- @param key="housing.corner_radius" min=0.0005 max=0.004

-- Core box (outer shell is everything within wall_r of it, z >= bottom).
local core_hx = 0.0175
local core_cy = -0.00405
local core_hy = 0.01745
local core_cz = 0.00165
local core_hz = 0.00415
local bottom = -0.0025

function is_inside(x, y, z)
    if z < bottom then
        return 0.0
    end
    -- Distance to the core box vs the corner radius.
    local qx = math.max(math.abs(x) - core_hx, 0.0)
    local qy = math.max(math.abs(y - core_cy) - core_hy, 0.0)
    local qz = math.max(math.abs(z - core_cz) - core_hz, 0.0)
    if qx * qx + qy * qy + qz * qz > wall_r * wall_r then
        return 0.0
    end
    -- Open-top JST notches (roof pulled back past the inner wall face).
    if x >= 0.0155 and y >= -0.0165 and y <= 0.0115 and z >= 0.0013 then
        return 0.0
    end
    if x <= -0.0155 and y >= -0.0145 and y <= 0.0065 and z >= 0.0013 then
        return 0.0
    end
    if y <= -0.0195 and x >= -0.011 and x <= 0.010 and z >= 0.0013 then
        return 0.0
    end
    -- North port band: USB mouth sized for the USB-IF maximum plug
    -- overmold (12.35 x 6.5mm) plus ~0.3mm clearance — the port is
    -- recessed ~2.6mm, so the whole wall opening must pass the overmold
    -- envelope. 13.0mm wide, open through the top, floor at z=0.15mm
    -- with r1.0 bottom corners; plus the flush-top scoop ledge.
    if y >= 0.0123 then
        local mx = math.max(math.abs(x) - 0.0055, 0.0)
        local mz = math.max(0.00115 - z, 0.0)
        if z >= 0.00015 and mx * mx + mz * mz <= 0.000001 then
            return 0.0
        end
        if x >= -0.007 and x <= 0.007 and z >= 0.0053 then
            return 0.0
        end
    end
    return 1.0
end

function get_bounds_min_x() return -0.0195 end
function get_bounds_max_x() return 0.0195 end
function get_bounds_min_y() return -0.0235 end
function get_bounds_max_y() return 0.0154 end
function get_bounds_min_z() return -0.0025 end
function get_bounds_max_z() return 0.0078 end
