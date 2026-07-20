-- Family D tray-side parting solid: everything below the seam, plus the
-- wall lip ring (3.1..4.3mm, inner face to wall midline), plus four snap
-- nub capsules half-proud on the lip outer faces.

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
    -- Lip ring: inside the midline rect, outside the inner-face rect.
    if z <= 0.0043 then
        if x >= -0.01855 and x <= 0.01855 and y >= -0.0225 and y <= 0.01432 then
            if not (x >= -0.0175 and x <= 0.0175 and y >= -0.0214 and y <= 0.01325) then
                return 1.0
            end
        end
    end
    -- Snap nubs: south pair on the lip outer face, north pair beside USB.
    if xcap(x, y, z, -0.0155, -0.0125, -0.0225, 0.0038, 0.0005) > 0.5 then
        return 1.0
    end
    if xcap(x, y, z, 0.0125, 0.0155, -0.0225, 0.0038, 0.0005) > 0.5 then
        return 1.0
    end
    if xcap(x, y, z, -0.0135, -0.0105, 0.01432, 0.0038, 0.0005) > 0.5 then
        return 1.0
    end
    if xcap(x, y, z, 0.0105, 0.0135, 0.01432, 0.0038, 0.0005) > 0.5 then
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
