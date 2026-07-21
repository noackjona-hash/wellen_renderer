struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    _padding: f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

// --- Constants ---
const PI: f32 = 3.14159265359;
const MAX_STEPS: i32 = 400;
const SURF_DIST: f32 = 0.01;
const MAX_DIST: f32 = 300.0;

// --- Sky & Atmosphere (Stormy, Overcast) ---
fn get_sky_color(ray_dir: vec3<f32>, sun_dir: vec3<f32>) -> vec3<f32> {
    // Stormy purplish-grey
    let v = pow(1.0 - max(ray_dir.y, 0.0), 2.0);
    let sky_base = mix(vec3<f32>(0.4, 0.35, 0.4), vec3<f32>(0.7, 0.65, 0.7), v);
    
    let sun_amount = max(dot(ray_dir, sun_dir), 0.0);
    let glow = pow(sun_amount, 8.0) * vec3<f32>(0.3, 0.3, 0.3);
    
    return sky_base + glow;
}

// --- Noise ---
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
    for (var i = 0; i < 4; i++) {
        value += amp * noise(p_mut);
        p_mut *= 2.0;
        amp *= 0.5;
    }
    return value;
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
    
    // Domain distortion: curve the wave forward (along -Z)
    var warped_p = p;
    let curve_strength = 0.25;
    warped_p.z += pow(max(p.y, 0.0), 1.5) * curve_strength;
    
    let wave_p = vec2<f32>(warped_p.x, warped_p.z);
    
    var h = 0.0;
    var shift = vec2<f32>(0.0);
    
    // Massive wave rolling towards the camera
    let w1 = Wave(vec2<f32>(0.0, -1.0), 1.0, 70.0, 1.0); // Big crest
    let w2 = Wave(vec2<f32>(0.2, -0.9), 0.5, 30.0, 1.2);
    let w3 = Wave(vec2<f32>(-0.3, -0.8), 0.3, 15.0, 1.3);
    let w4 = Wave(vec2<f32>(0.5, -0.5), 0.2, 5.0, 1.5);
    
    let g1 = gerstner_wave(wave_p, w1, time);
    shift += g1.xz; h += g1.y;
    let g2 = gerstner_wave(wave_p + shift, w2, time);
    shift += g2.xz; h += g2.y;
    let g3 = gerstner_wave(wave_p + shift, w3, time);
    shift += g3.xz; h += g3.y;
    let g4 = gerstner_wave(wave_p + shift, w4, time);
    h += g4.y;
    
    // Turbulence & micro-waves
    h += (fbm(wave_p * 0.8) - 0.5) * 1.5;
    
    return warped_p.y - h;
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>) -> f32 {
    var dO = 0.0;
    for(var i=0; i<MAX_STEPS; i++) {
        let p = ro + rd * dO;
        let dS = map(p);
        
        // Small step size to handle heavy domain distortion safely
        dO += dS * 0.2; 
        
        if (abs(dS) < SURF_DIST || dO > MAX_DIST) {
            break;
        }
    }
    return dO;
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.05, 0.0);
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
    
    // Camera is very low, looking straight at the face of the breaking wave
    let ro = vec3<f32>(0.0, 2.0, 30.0);
    let lookat = vec3<f32>(0.0, 10.0, 0.0); // Look at the crest
    let f = normalize(lookat - ro);
    let r = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), f));
    let u = cross(f, r);
    // Wide angle FOV to make it look massive
    let rd = normalize(coord.x * r + coord.y * u + 0.7 * f);
    
    // Light coming from behind the wave to give that emerald SSS glow
    let sun_dir = normalize(vec3<f32>(0.2, 0.4, -0.8)); 
    
    var col = vec3<f32>(0.0);
    
    let d = raymarch(ro, rd);
    if (d < MAX_DIST) {
        let p = ro + rd * d;
        let n = calc_normal(p);
        let v = -rd;
        
        // Deep emerald green ocean
        let water_base = vec3<f32>(0.02, 0.15, 0.12);
        
        // Vibrant emerald for the thin crest
        // Using height and normal facing the sun
        let crest_mask = smoothstep(2.0, 20.0, p.y) * max(-n.z, 0.0);
        let water_shallow = vec3<f32>(0.1, 0.8, 0.5); 
        let water_col = mix(water_base, water_shallow, crest_mask);
        
        let l = sun_dir;
        let h_vec = normalize(v + l);
        let ndotl = max(dot(n, l), 0.0);
        let ndotv = max(dot(n, v), 0.0);
        
        // Specular
        let roughness = 0.2;
        let a2 = roughness * roughness;
        let ndoth = max(dot(n, h_vec), 0.0);
        let denom = (ndoth * ndoth * (a2 - 1.0) + 1.0);
        let spec = (a2 / (PI * denom * denom)) * vec3<f32>(0.8, 0.9, 0.9) * ndotl * 0.3;
        
        // Reflection
        let ref_dir = reflect(rd, n);
        let sky_col = get_sky_color(ref_dir, sun_dir);
        let f0 = 0.02;
        let f_val = fresnel(ndotv, f0);
        
        // Foam!
        // Appears on high slopes, peaks, and turbulence
        let slope = 1.0 - max(n.y, 0.0); // 0 at flat, 1 at vertical
        var foam_mask = smoothstep(0.4, 1.0, slope) + smoothstep(12.0, 20.0, p.y);
        
        // Break up foam with FBM
        let foam_noise = fbm(vec2<f32>(p.x, p.y + p.z) * 3.0);
        foam_mask *= foam_noise * 1.5;
        foam_mask = clamp(foam_mask, 0.0, 1.0);
        
        // Highlight foam where it hits the sun
        let foam_col = mix(vec3<f32>(0.7, 0.75, 0.8), vec3<f32>(1.0, 1.0, 1.0), ndotl);
        
        // SSS - massive glow from behind
        let sss_amount = max(dot(rd, sun_dir), 0.0);
        let sss = pow(sss_amount, 2.0) * vec3<f32>(0.2, 1.0, 0.5) * crest_mask * 0.8;
        
        // Combine
        var surface_col = mix(water_col, sky_col, f_val) + spec + sss;
        col = mix(surface_col, foam_col, foam_mask);
        
        // Distance Fog
        let fog = 1.0 - exp(-d * 0.003);
        col = mix(col, get_sky_color(rd, sun_dir), fog);
        
    } else {
        col = get_sky_color(rd, sun_dir);
    }
    
    // Tonemapping & Gamma
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    
    textureStore(out_tex, vec2<i32>(global_id.xy), vec4<f32>(col, 1.0));
}
