-- Family D lid-side cut solid: everything below the seam, plus the rebate
-- ring 0.2mm larger than the tray lip on every face, plus groove capsules
-- matching the snap nubs (r +0.15mm, ends +0.2mm).

local seam = 0.0031

function xcap(px, py, pz, x0, x1, cy, cz, r)
    local sx = math.min(math.max(px, x0), x1)
    local dx = px - sx
    local dy = py - cy
    local dz = pz - cz
    if dx * dx + dy * dy + dz * dz <= r * r then
        return 1.0
    end
    return 0.0
end

function is_inside(x, y, z)
    if z <= seam then
        return 1.0
    end
    -- Rebate ring (lip + 0.2mm clearance on every face, taller top).
    if z <= 0.0045 then
        if x >= -0.01875 and x <= 0.01875 and y >= -0.0227 and y <= 0.01452 then
            if not (x >= -0.0173 and x <= 0.0173 and y >= -0.0212 and y <= 0.01305) then
                return 1.0
            end
        end
    end
    -- Grooves for the snap nubs.
    if xcap(x, y, z, -0.0157, -0.0123, -0.0225, 0.0038, 0.00065) > 0.5 then
        return 1.0
    end
    if xcap(x, y, z, 0.0123, 0.0157, -0.0225, 0.0038, 0.00065) > 0.5 then
        return 1.0
    end
    if xcap(x, y, z, -0.0137, -0.0103, 0.01432, 0.0038, 0.00065) > 0.5 then
        return 1.0
    end
    if xcap(x, y, z, 0.0103, 0.0137, 0.01432, 0.0038, 0.00065) > 0.5 then
        return 1.0
    end
    return 0.0
end

function get_bounds_min_x() return -0.03 end
function get_bounds_max_x() return 0.03 end
function get_bounds_min_y() return -0.03 end
function get_bounds_max_y() return 0.03 end
function get_bounds_min_z() return -0.01 end
function get_bounds_max_z() return 0.005 end
