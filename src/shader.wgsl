struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

// --- Constants ---
const PI: f32 = 3.14159265359;
const MAX_STEPS: i32 = 300;
const SURF_DIST: f32 = 0.01;
const MAX_DIST: f32 = 500.0;

// --- Sky & Atmosphere ---
fn get_sky_color(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    let sun_amount = max(dot(ray_dir, sun_dir), 0.0);
    
    // Horizon glow (pinkish orange for sunrise)
    let v = pow(1.0 - max(ray_dir.y, 0.0), 3.0);
    let sky_base = mix(vec3<f32>(0.05, 0.1, 0.3), vec3<f32>(0.9, 0.4, 0.3), v);
    
    // Sun disc
    var sun = 0.0;
    if (sun_amount > 0.9995) {
        sun = 2.0; // intense center
    }
    
    // Sun glow (warm yellow/orange)
    let glow = pow(sun_amount, 64.0) * vec3<f32>(1.0, 0.6, 0.2) * 2.0 + 
               pow(sun_amount, 8.0) * vec3<f32>(1.0, 0.3, 0.1) * 0.5;
    
    return sky_base + sun * vec3<f32>(1.0, 1.0, 0.9) + glow;
}

// --- Waves ---
struct Wave {
    dir: vec2<f32>,
    steepness: f32,
    wavelength: f32,
    speed: f32,
}

fn gerstner_wave(pos: vec2<f32>, wave: Wave, time: f32) -> vec3<f32> {
    let k = 2.0 * PI / wave.wavelength;
    let c = sqrt(9.8 / k) * wave.speed;
    let d = normalize(wave.dir);
    let f = k * (dot(d, pos) - c * time);
    let a = wave.steepness / k;
    
    return vec3<f32>(
        d.x * a * cos(f),
        a * sin(f),
        d.y * a * cos(f)
    );
}

fn map(p: vec3<f32>) -> f32 {
    let time = uniforms.time;
    let wave_p = vec2<f32>(p.x, p.z);
    
    var h = 0.0;
    var shift = vec2<f32>(0.0);
    
    // Multiple octaves of waves for hyper-realism
    // Using 6 octaves of varying frequencies and directions
    let w1 = Wave(vec2<f32>(1.0, 0.3), 0.35, 30.0, 1.0);
    let w2 = Wave(vec2<f32>(0.7, -0.6), 0.25, 18.0, 1.1);
    let w3 = Wave(vec2<f32>(-0.4, 0.9), 0.2, 10.0, 1.2);
    let w4 = Wave(vec2<f32>(-0.8, -0.3), 0.15, 5.0, 1.3);
    let w5 = Wave(vec2<f32>(0.5, 0.1), 0.1, 2.5, 1.5);
    let w6 = Wave(vec2<f32>(-0.1, -0.7), 0.05, 1.2, 1.8);
    let w7 = Wave(vec2<f32>(0.3, 0.4), 0.03, 0.5, 2.0); // very high freq details
    
    let g1 = gerstner_wave(wave_p, w1, time);
    shift += g1.xz; h += g1.y;
    let g2 = gerstner_wave(wave_p + shift, w2, time);
    shift += g2.xz; h += g2.y;
    let g3 = gerstner_wave(wave_p + shift, w3, time);
    shift += g3.xz; h += g3.y;
    let g4 = gerstner_wave(wave_p + shift, w4, time);
    shift += g4.xz; h += g4.y;
    let g5 = gerstner_wave(wave_p + shift, w5, time);
    shift += g5.xz; h += g5.y;
    let g6 = gerstner_wave(wave_p + shift, w6, time);
    shift += g6.xz; h += g6.y;
    let g7 = gerstner_wave(wave_p + shift, w7, time);
    h += g7.y;
    
    // Base height of water is at y = 0
    return p.y - h;
}

// Raymarching
fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var dO = 0.0;
    for(var i=0; i<MAX_STEPS; i++) {
        let p = ro + rd * dO;
        let dS = map(p);
        
        // Relaxed stepping to avoid overshooting steep Gerstner waves
        dO += dS * 0.6;
        
        if (abs(dS) < SURF_DIST || dO > MAX_DIST) {
            break;
        }
    }
    return dO;
}

// Calculate Normal using central differences
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.02, 0.0);
    let n = vec3<f32>(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    );
    return normalize(n);
}

// Fresnel Schlick
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
    
    // Animated Camera Setup (moving slowly forward)
    let cam_speed = 3.0;
    let ro = vec3<f32>(0.0, 6.0 + sin(uniforms.time*0.5)*0.5, uniforms.time * cam_speed);
    let lookat = vec3<f32>(0.0, 2.0, ro.z + 10.0);
    let f = normalize(lookat - ro);
    let r = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), f));
    let u = cross(f, r);
    let rd = normalize(coord.x * r + coord.y * u + 1.2 * f);
    
    // Sunrise position (low on horizon, slightly to the right)
    let sun_dir = normalize(vec3<f32>(0.6, 0.1, 1.0)); 
    
    var col = vec3<f32>(0.0);
    
    let d = raymarch(ro, rd);
    if (d < MAX_DIST) {
        let p = ro + rd * d;
        let n = calc_normal(p);
        let v = -rd;
        
        // Deep ocean blue-green base
        let water_base = vec3<f32>(0.01, 0.04, 0.12);
        
        // The peak of the wave gets a brighter, more transmissive color
        let peak_factor = smoothstep(-1.0, 2.0, p.y); 
        let water_shallow = vec3<f32>(0.05, 0.25, 0.25);
        let water_col = mix(water_base, water_shallow, peak_factor);
        
        let l = sun_dir;
        let h = normalize(v + l);
        let ndotl = max(dot(n, l), 0.0);
        let ndotv = max(dot(n, v), 0.0);
        
        // Specular (Sun reflection) using GGX-like distribution
        let roughness = 0.1;
        let a2 = roughness * roughness;
        let ndoth = max(dot(n, h), 0.0);
        let denom = (ndoth * ndoth * (a2 - 1.0) + 1.0);
        let spec_brdf = a2 / (PI * denom * denom);
        
        // Intense specular highlight from the sun
        let spec = spec_brdf * vec3<f32>(1.0, 0.7, 0.3) * ndotl * 3.0;
        
        // Reflection of the sky
        // Shift reflection normal slightly to simulate microsurface roughness
        let ref_dir = reflect(rd, n);
        let sky_col = get_sky_color(ref_dir, sun_dir);
        
        // Fresnel
        let f0 = 0.02; // Water IOR ~ 1.33
        let f_val = fresnel(ndotv, f0);
        
        // Subsurface scattering approximation (sun shining through wave peaks)
        let sss_amount = max(dot(rd, sun_dir), 0.0);
        let sss = pow(sss_amount, 6.0) * vec3<f32>(0.8, 0.9, 0.6) * max(p.y, 0.0) * 0.15;
        
        // Combine lighting
        col = mix(water_col, sky_col, f_val) + spec + sss;
        
        // Distance Fog (Atmospheric scattering)
        let fog = 1.0 - exp(-d * 0.001);
        let fog_col = get_sky_color(rd, sun_dir);
        col = mix(col, fog_col, fog);
        
    } else {
        col = get_sky_color(rd, sun_dir);
    }
    
    // Tonemapping (ACES approximation)
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    
    // Gamma correction
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    
    textureStore(out_tex, vec2<i32>(global_id.xy), vec4<f32>(col, 1.0));
}
