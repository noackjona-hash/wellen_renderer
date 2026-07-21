struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

const PI: f32 = 3.14159265359;
const MAX_STEPS: i32 = 400;
const SURF_DIST: f32 = 0.01;
const MAX_DIST: f32 = 300.0;

fn get_sky_color(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    // Gloomy, overcast grey sky with a hint of purple
    let v = pow(1.0 - max(ray_dir.y, 0.0), 1.2);
    let sky_base = mix(vec3<f32>(0.5, 0.48, 0.52), vec3<f32>(0.75, 0.75, 0.8), v);
    
    // Very diffuse sun behind heavy clouds
    let sun_amount = max(dot(ray_dir, sun_dir), 0.0);
    let glow = pow(sun_amount, 4.0) * vec3<f32>(0.3, 0.3, 0.35);
    
    return sky_base + glow;
}

// --- Noise Functions ---
fn hash(p: vec2<f32>) -> f32 {
    var p2 = fract(p * vec2<f32>(123.34, 456.21));
    p2 += dot(p2, p2 + 45.32);
    return fract(p2.x * p2.y);
}

fn noise(x: vec2<f32>) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(p + vec2<f32>(0.0,0.0)), hash(p + vec2<f32>(1.0,0.0)), u.x),
               mix(hash(p + vec2<f32>(0.0,1.0)), hash(p + vec2<f32>(1.0,1.0)), u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
    var value = 0.0;
    var amp = 0.5;
    var p_mut = p;
    for (var i = 0; i < 6; i++) {
        value += amp * noise(p_mut);
        p_mut *= 2.0;
        amp *= 0.5;
    }
    return value;
}

// Ridge noise for sharp, choppy water
fn ridge_noise(p: vec2<f32>) -> f32 {
    var n = fbm(p);
    return 1.0 - abs(n * 2.0 - 1.0); // Creates sharp ridges
}

// --- Giant Wave Shape ---
fn map(p: vec3<f32>) -> f32 {
    var warped_p = p;
    
    let wave_height = 28.0;
    let wave_width = 8.0; // Much narrower, sharper peak
    
    // Curl: Shift Z heavily based on Y
    let curl_strength = 22.0;
    let curl = smoothstep(5.0, 30.0, p.y) * curl_strength;
    warped_p.z += curl;
    
    // We also want the wave to bowl (curve inwards in X)
    let bowl = smoothstep(0.0, 40.0, abs(p.x)) * 8.0;
    warped_p.z -= bowl;
    
    let w_z = warped_p.z;
    let w_x = warped_p.x;
    
    // Use an exponential decay for a sharp, jagged peak instead of smooth Gaussian
    var h = exp(-abs(w_z) / wave_width) * wave_height;
    
    // Make the back of the wave (w_z > 0) smooth and long, 
    // and the front (w_z < 0) vertical and steep
    if (w_z > 0.0) {
        h = exp(-w_z / (wave_width * 2.0)) * wave_height; 
    } else {
        h = exp(w_z / (wave_width * 0.7)) * wave_height;
    }
    
    // Deep trough in front of the wave
    if (w_z < -5.0 && w_z > -20.0) {
        h -= sin((w_z + 5.0) * 0.2) * 2.0; 
    }
    
    // Taper off horizontally
    h *= exp(-(w_x * w_x) / 1000.0);
    
    // Aggressive chop on the surface
    var chop = ridge_noise(vec2<f32>(w_x, warped_p.z) * 0.2) * 2.0;
    chop += ridge_noise(vec2<f32>(w_x, warped_p.z) * 0.8) * 0.5;
    
    // Reduce chop at the very crest so it looks like it's stretching thin
    let stretch = smoothstep(15.0, 28.0, p.y);
    chop = mix(chop, 0.0, stretch * 0.8);
    
    h += chop;
    
    // Base ocean level is slightly restless
    let base_ocean = (fbm(p.xz * 0.1) - 0.5) * 3.0;
    
    return p.y - max(h, base_ocean);
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var dO = 0.0;
    for(var i=0; i<MAX_STEPS; i++) {
        let p = ro + rd * dO;
        let dS = map(p);
        
        // Very conservative step because of the sharp peak and curl
        dO += dS * 0.1; 
        
        if (abs(dS) < SURF_DIST || dO > MAX_DIST) {
            break;
        }
    }
    return dO;
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.02, 0.0);
    let n = vec3<f32>(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    );
    return normalize(n);
}

fn fresnel(cos_theta: f32, f0: f32) -> f32 {
    return f0 + (1.0 - f0) * pow(max(1.0 - cos_theta, 0.0), 5.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let res = uniforms.resolution;
    let fc = vec2<f32>(f32(global_id.x), f32(global_id.y));
    
    if (fc.x >= res.x || fc.y >= res.y) {
        return;
    }
    
    let uv = (fc * 2.0 - res) / res.y;
    let coord = vec2<f32>(uv.x, -uv.y);
    
    // Position camera low, close to the water, looking at the crashing face
    let ro = vec3<f32>(0.0, 1.0, 35.0);
    let lookat = vec3<f32>(0.0, 10.0, 0.0); 
    let f = normalize(lookat - ro);
    let r = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), f));
    let u = cross(f, r);
    // Standard FOV
    let rd = normalize(coord.x * r + coord.y * u + 0.8 * f);
    
    // Light is coming from behind and slightly above the wave
    let sun_dir = normalize(vec3<f32>(0.0, 0.2, -1.0)); 
    
    var col = vec3<f32>(0.0);
    
    let d = raymarch(ro, rd);
    if (d < MAX_DIST) {
        let p = ro + rd * d;
        let n = calc_normal(p);
        let v = -rd;
        
        // Deep teal base
        let water_base = vec3<f32>(0.0, 0.05, 0.06);
        
        // Vibrant emerald green subsurface glow
        // Intense where normal faces away from light, and high up
        let thickness = smoothstep(28.0, 5.0, p.y); // thinner at the top
        let sss_mask = smoothstep(0.0, 1.0, dot(rd, sun_dir)) * (1.0 - thickness) * max(-n.z, 0.0);
        
        let sss_col = vec3<f32>(0.0, 0.7, 0.35); // Emerald green
        let water_col = mix(water_base, sss_col, sss_mask * 1.5);
        
        // Reflections
        let ref_dir = reflect(rd, n);
        let sky_col = get_sky_color(ref_dir, sun_dir);
        let f0 = 0.02;
        let f_val = fresnel(max(dot(n, v), 0.0), f0);
        
        var surface_col = mix(water_col, sky_col, f_val * 0.4);
        
        // --- FOAM GENERATION ---
        
        // 1. Foam at the crest (thick white lip)
        let crest_proximity = smoothstep(22.0, 28.0, p.y);
        var crest_foam = crest_proximity * (fbm(p.xz * 5.0) * 0.5 + 0.5);
        
        // 2. Cascading streaks down the face
        let face_slope = max(-n.z, 0.0);
        // Stretch noise along Y to create vertical streaks
        let streak_uv = vec2<f32>(p.x * 1.5, p.y * 0.1 + p.z * 1.0);
        let streak_noise = smoothstep(0.4, 0.8, fbm(streak_uv));
        let cascade_foam = smoothstep(5.0, 24.0, p.y) * face_slope * streak_noise * 1.5;
        
        // 3. Webbing/Marbling (Voronoi-like)
        let web_uv = vec2<f32>(p.x * 0.5, p.y * 0.5 + p.z * 0.5);
        let web_noise = ridge_noise(web_uv);
        let webbing = smoothstep(0.6, 1.0, web_noise) * face_slope * smoothstep(2.0, 20.0, p.y) * 0.5;
        
        var foam_mask = crest_foam + cascade_foam + webbing;
        foam_mask = clamp(foam_mask, 0.0, 1.0);
        
        // White foam with slight blue tint in shadows
        let foam_light = mix(vec3<f32>(0.6, 0.7, 0.75), vec3<f32>(1.0, 1.0, 1.0), max(dot(n, sun_dir), 0.0) * 0.5 + 0.5);
        
        col = mix(surface_col, foam_light, foam_mask);
        
        // Mist/Spray coming off the wave
        let mist = exp(-d * 0.005) * smoothstep(15.0, 28.0, p.y) * 0.15;
        col = mix(col, vec3<f32>(0.8, 0.8, 0.85), mist);
        
    } else {
        col = get_sky_color(rd, sun_dir);
    }
    
    // Tonemapping & Contrast
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    
    textureStore(out_tex, vec2<i32>(global_id.xy), vec4<f32>(col, 1.0));
}
