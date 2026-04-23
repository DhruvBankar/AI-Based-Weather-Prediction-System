/**
 * ============================================================
 *  PREMIUM WEATHER SYSTEM — Production Grade
 *  All rendering done on Canvas for true 60fps with no CSS hacks.
 *  Architecture:
 *    WeatherSystem      — top-level controller & rAF loop
 *    StateController    — manages lerp between weather states
 *    SkyRenderer        — draws sky gradient + clouds + stars
 *    RainEngine         — Canvas2D particle rain system
 *    MouseParallax      — smooth mouse-reactive layer offsets
 * ============================================================
 */

'use strict';

/* ====================================================================
   1.  CONSTANTS & STATE DEFINITIONS
   ==================================================================== */

/** Smooth easing: ease-in-out cubic */
function easeInOut(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/** Linear interpolation */
function lerp(a, b, t) { return a + (b - a) * t; }

/** Interpolate between two [r,g,b] arrays INTO an existing array (avoids GC allocation) */
function lerpRGBInto(c1, c2, t, out) {
    out[0] = Math.round(lerp(c1[0], c2[0], t));
    out[1] = Math.round(lerp(c1[1], c2[1], t));
    out[2] = Math.round(lerp(c1[2], c2[2], t));
}
/** Interpolate between two [r,g,b] arrays — kept for any one-off use */
function lerpRGB(c1, c2, t) {
    return [
        Math.round(lerp(c1[0], c2[0], t)),
        Math.round(lerp(c1[1], c2[1], t)),
        Math.round(lerp(c1[2], c2[2], t))
    ];
}

/** Convert [r,g,b] to css string */
function rgb(c) { return `rgb(${c[0]},${c[1]},${c[2]})`; }
function rgba(c, a) { return `rgba(${c[0]},${c[1]},${c[2]},${a})`; }

/**
 * Each state defines:
 *  skyTop / skyBot     — gradient colors as [r,g,b]
 *  sunColor            — atmospheric glow color [r,g,b]
 *  sunAlpha            — how strong the glow is (0-1)
 *  cloudAlpha          — base opacity of cloud layers
 *  skyMid               — [r,g,b] middle gradient stop (adds 3rd realism layer)
 *  cloudColor          — [r,g,b] tint of clouds (warm at sunset)
 *  fogAlpha            — bottom fog density
 *  starAlpha           — star layer visibility
 *  rainIntensity       — 0 = off, 1 = full rain
 *  sunY                — normalized Y position of sun glow (0=top, 1=bottom)
 */
const STATES = {
    SUNNY: {
        skyTop:   [32, 110, 195],
        skyMid:   [72, 158, 228],
        skyBot:   [135, 206, 250],
        sunColor: [255, 245, 200],
        sunAlpha: 0.15,
        sunY:     0.30,
        cloudAlpha:  [0.45, 0.75, 0.95],
        cloudColor:  [255, 255, 255],
        fogAlpha: 0,
        starAlpha: 0,
        rainIntensity: 0
    },
    SUNSET: {
        // Real sunset: deep indigo zenith → coral/mauve mid-sky → warm amber horizon
        skyTop:   [14, 16, 48],       // deep blue-indigo at zenith
        skyMid:   [185, 72, 85],      // warm coral/mauve band in upper-mid sky
        skyBot:   [255, 145, 42],     // amber-gold near horizon
        sunColor: [255, 195, 100],    // golden glow near the horizon
        sunAlpha: 0.60,               // strong: sun is visible on horizon
        sunY:     0.78,               // placed low, near the horizon
        cloudAlpha:  [0.22, 0.45, 0.65],
        cloudColor:  [255, 175, 110], // warm golden-amber lit clouds
        fogAlpha: 0.08,
        starAlpha: 0.15,              // first stars barely visible
        rainIntensity: 0
    },
    NIGHT: {
        skyTop:   [5, 8, 20],
        skyMid:   [8, 16, 38],
        skyBot:   [12, 25, 55],
        sunColor: [80, 100, 180],
        sunAlpha: 0.04,
        sunY:     0.30,
        cloudAlpha:  [0.08, 0.15, 0.25],
        cloudColor:  [140, 155, 200],
        fogAlpha: 0,
        starAlpha: 1.0,
        rainIntensity: 0
    },
    RAIN: {
        skyTop:   [55, 65, 80],
        skyMid:   [72, 82, 97],
        skyBot:   [90, 100, 115],
        sunColor: [160, 180, 200],
        sunAlpha: 0.0,
        sunY:     0.30,
        cloudAlpha:  [0.6, 0.9, 1.0],
        cloudColor:  [200, 210, 225],
        fogAlpha: 0.55,
        starAlpha: 0,
        rainIntensity: 1.0
    }
};


/* ====================================================================
   2.  MOUSE PARALLAX ENGINE
   ==================================================================== */
class MouseParallax {
    constructor() {
        this.target = { x: 0, y: 0 };
        this.current = { x: 0, y: 0 };
        this._ticking = false;
        this._bound = (e) => {
            if (!this._ticking) {
                requestAnimationFrame(() => {
                    this._onMove(e);
                    this._ticking = false;
                });
                this._ticking = true;
            }
        };
        window.addEventListener('mousemove', this._bound, { passive: true });
    }

    _onMove(e) {
        // Normalize to -1 … +1
        this.target.x = (e.clientX / window.innerWidth  - 0.5) * 2;
        this.target.y = (e.clientY / window.innerHeight - 0.5) * 2;
    }

    /** Call every frame with actual dt (seconds) for frame-rate independence */
    update(dt) {
        // Use dt-based lerp factor so it's identical at 30fps and 120fps
        const k = 1 - Math.pow(0.04, dt); // ~96% per second decay
        this.current.x += (this.target.x - this.current.x) * k;
        this.current.y += (this.target.y - this.current.y) * k;
    }

    /** Get pixel offset for a layer at given depth (0=far, 1=near) */
    getOffset(depth) {
        const maxPx = lerp(8, 28, depth);
        return {
            x: this.current.x * maxPx,
            y: this.current.y * maxPx * 0.5
        };
    }
}


/* ====================================================================
   3.  SKY RENDERER (Canvas 2D)
   ==================================================================== */
class SkyRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx    = canvas.getContext('2d');

        // Cloud layer offsets for parallax scrolling (pixels)
        this.cloudOffsets = [0, 0, 0]; // far, mid, near

        // Stars: generate once
        this.stars = this._generateStars(120);
        this.starPhases = this.stars.map(() => Math.random() * Math.PI * 2);

        this.resize();
        window.addEventListener('resize', () => this.resize(), { passive: true });
    }

    resize() {
        this.W = this.canvas.width  = window.innerWidth;
        this.H = this.canvas.height = window.innerHeight;
    }

    _generateStars(count) {
        const stars = [];
        for (let i = 0; i < count; i++) {
            stars.push({
                x: Math.random(),      // normalized 0-1
                y: Math.random() * 0.75, // upper 75% of sky
                r: 0.5 + Math.random() * 1.5,
                speed: 0.5 + Math.random() * 1.5  // twinkle speed Hz
            });
        }
        return stars;
    }

    /**
     * Draw a single cloud puff at (cx, cy) with given radius and color.
     * Uses stacked ellipses for a billowy look.
     */
    _drawCloudPuff(ctx, cx, cy, rx, ry, color, alpha) {
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = color;

        const bumps = [
            [0,    0,    1,   1],
            [-0.6, 0.2,  0.7, 0.7],
            [0.6,  0.2,  0.7, 0.7],
            [-0.35,-0.3, 0.6, 0.6],
            [0.35, -0.3, 0.6, 0.6],
        ];
        for (const [dx, dy, sx, sy] of bumps) {
            ctx.beginPath();
            ctx.ellipse(cx + dx * rx, cy + dy * ry, rx * sx, ry * sy, 0, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    }

    /** Draw a full cloud formation (cluster of puffs) */
    _drawCloud(ctx, x, y, scale, color, alpha) {
        const rx = 55 * scale, ry = 28 * scale;
        this._drawCloudPuff(ctx, x,          y,      rx,     ry,      color, alpha);
        this._drawCloudPuff(ctx, x + rx*0.9, y+6,   rx*0.7, ry*0.7,  color, alpha * 0.9);
        this._drawCloudPuff(ctx, x - rx*0.8, y+8,   rx*0.6, ry*0.6,  color, alpha * 0.85);
    }

    /**
     * Draw one cloud layer using seamless looping.
     * clouds = array of {relX, relY, scale, alpha}  (pre-computed per layer)
     */
    _drawCloudLayer(ctx, clouds, offsetX, baseAlpha, cloudColor, blur) {
        const cStr = rgb(cloudColor);
        if (blur > 0) {
            ctx.filter = `blur(${blur}px)`;
        }
        for (const c of clouds) {
            // Wrap position seamlessly using modulo across 2× screen width
            const x = ((c.relX * this.W + offsetX) % (this.W * 2) + this.W * 2) % (this.W * 2) - this.W * 0.5;
            const y = c.relY * this.H;
            this._drawCloud(ctx, x, y, c.scale, cStr, c.alpha * baseAlpha);
        }
        ctx.filter = 'none';
    }

    /** Main draw call — called every frame */
    draw(state, time, mouseParallax) {
        const { ctx, W, H } = this;
        ctx.clearRect(0, 0, W, H);

        /* --- SKY GRADIENT (3-stop for realism) --- */
        const grad = ctx.createLinearGradient(0, 0, 0, H);
        grad.addColorStop(0,    rgb(state.skyTop));
        grad.addColorStop(0.42, rgb(state.skyMid));  // mid-sky colour band
        grad.addColorStop(1,    rgb(state.skyBot));
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H);

        /* --- ATMOSPHERIC SUN / HORIZON GLOW --- */
        if (state.sunAlpha > 0.01) {
            // sunY controls where the glow sits: high for daytime, low for sunset
            const gx = W * 0.52, gy = H * state.sunY;
            const glowGrad = ctx.createRadialGradient(gx, gy, 0, gx, gy, W * 0.60);
            glowGrad.addColorStop(0,   rgba(state.sunColor, state.sunAlpha));
            glowGrad.addColorStop(0.4, rgba(state.sunColor, state.sunAlpha * 0.4));
            glowGrad.addColorStop(1,   rgba(state.sunColor, 0));
            ctx.fillStyle = glowGrad;
            ctx.fillRect(0, 0, W, H);
        }

        /* --- STARS --- */
        if (state.starAlpha > 0.01) {
            // Batch stars by similar alpha to reduce fillStyle changes.
            // Group into 8 alpha buckets (0.0–1.0 in 0.125 steps) to minimize state switches.
            for (let i = 0; i < this.stars.length; i++) {
                const s = this.stars[i];
                // Smooth twinkle: sine wave, frequency varies per star
                const twinkle = 0.5 + 0.5 * Math.sin(time * s.speed + this.starPhases[i]);
                const alpha = state.starAlpha * (0.4 + 0.6 * twinkle);
                // Quantize alpha to 16 steps to enable browser canvas batching
                const alphaQ = Math.round(alpha * 16) / 16;
                ctx.fillStyle = `rgba(255,255,255,${alphaQ})`;
                ctx.beginPath();
                ctx.arc(s.x * W, s.y * H, s.r * (0.85 + 0.15 * twinkle), 0, Math.PI * 2);
                ctx.fill();
            }
        }

        /* --- CLOUDS (3 layers with depth) --- */
        // Far layer
        const farOffset = mouseParallax.getOffset(0);
        this._drawCloudLayer(
            ctx, this._cloudsFar,
            this.cloudOffsets[0] + farOffset.x,
            state.cloudAlpha[0],
            state.cloudColor, 2
        );

        // Mid layer — 0.5px blur removed: sub-pixel blur triggers full GPU filter
        // pipeline with zero visible improvement. Far layer's 2px blur gives
        // enough depth illusion by contrast.
        const midOffset = mouseParallax.getOffset(0.5);
        this._drawCloudLayer(
            ctx, this._cloudsMid,
            this.cloudOffsets[1] + midOffset.x,
            state.cloudAlpha[1],
            state.cloudColor, 0
        );

        // Near layer
        const nearOffset = mouseParallax.getOffset(1);
        this._drawCloudLayer(
            ctx, this._cloudsNear,
            this.cloudOffsets[2] + nearOffset.x,
            state.cloudAlpha[2],
            state.cloudColor, 0
        );

        /* --- FOG LAYER --- */
        if (state.fogAlpha > 0.01) {
            const fogGrad = ctx.createLinearGradient(0, H * 0.6, 0, H);
            fogGrad.addColorStop(0, `rgba(200,210,220,0)`);
            fogGrad.addColorStop(1, `rgba(200,210,220,${state.fogAlpha})`);
            ctx.fillStyle = fogGrad;
            ctx.fillRect(0, H * 0.6, W, H * 0.4);
        }
    }

    /** Pre-compute cloud positions for each layer (done once, not every frame) */
    initClouds() {
        // Far clouds — smaller, higher, more of them
        this._cloudsFar = Array.from({ length: 8 }, (_, i) => ({
            relX: (i / 7) * 2 - 0.1, // spread across 0…2× screen width
            relY: 0.1 + Math.random() * 0.2,
            scale: 0.5 + Math.random() * 0.4,
            alpha: 0.4 + Math.random() * 0.3
        }));
        // Mid clouds
        this._cloudsMid = Array.from({ length: 6 }, (_, i) => ({
            relX: (i / 5) * 2 - 0.1,
            relY: 0.25 + Math.random() * 0.2,
            scale: 0.8 + Math.random() * 0.5,
            alpha: 0.5 + Math.random() * 0.3
        }));
        // Near clouds — large, lower
        this._cloudsNear = Array.from({ length: 4 }, (_, i) => ({
            relX: (i / 3) * 2 - 0.1,
            relY: 0.45 + Math.random() * 0.15,
            scale: 1.2 + Math.random() * 0.6,
            alpha: 0.55 + Math.random() * 0.3
        }));
    }

    /** Advance cloud scroll offsets — call every frame with dt */
    updateClouds(dt) {
        // Pixels per second for each layer (far = slow, near = fast)
        this.cloudOffsets[0] += 12 * dt;   // far
        this.cloudOffsets[1] += 28 * dt;   // mid
        this.cloudOffsets[2] += 55 * dt;   // near
    }
}


/* ====================================================================
   4.  RAIN ENGINE (Canvas 2D)
   ==================================================================== */
class RainEngine {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx    = canvas.getContext('2d');
        this.drops  = [];
        this._intensity = 0; // 0-1 interpolated
        this._targetIntensity = 0;

        this.resize();
        window.addEventListener('resize', () => this.resize(), { passive: true });
        this._populateDrops(150);
    }

    resize() {
        this.W = this.canvas.width  = window.innerWidth;
        this.H = this.canvas.height = window.innerHeight;
    }

    _populateDrops(count) {
        for (let i = 0; i < count; i++) {
            this.drops.push(this._newDrop(true));
        }
    }

    _newDrop(randomY = false) {
        return {
            x:       Math.random() * this.W,
            y:       randomY ? Math.random() * this.H : -20,
            speed:   400 + Math.random() * 300,    // px/s
            length:  12 + Math.random() * 22,
            opacity: 0.15 + Math.random() * 0.45,
            wind:    80 + Math.random() * 60        // horizontal px/s (right tilt)
        };
    }

    setIntensity(target) {
        this._targetIntensity = target;
        // Only write to DOM when opacity bucket changes (avoids style recalc every frame)
        const opacityVal = target > 0 ? 1 : 0;
        if (this._lastOpacity !== opacityVal) {
            this._lastOpacity = opacityVal;
            this.canvas.style.opacity = opacityVal;
        }
    }

    draw(dt) {
        // Lerp rendered intensity
        const k = 1 - Math.pow(0.001, dt);
        this._intensity += (this._targetIntensity - this._intensity) * k;

        if (this._intensity < 0.01) {
            this.ctx.clearRect(0, 0, this.W, this.H);
            return;
        }

        this.ctx.clearRect(0, 0, this.W, this.H);
        this.ctx.lineWidth = 1;
        this.ctx.lineCap   = 'round';

        const count = Math.floor(this.drops.length * this._intensity);
        for (let i = 0; i < count; i++) {
            const p = this.drops[i];

            this.ctx.globalAlpha = p.opacity * this._intensity;
            this.ctx.strokeStyle = 'rgba(200, 220, 255, 1)';
            this.ctx.beginPath();
            this.ctx.moveTo(p.x, p.y);
            this.ctx.lineTo(p.x + p.wind * 0.07, p.y + p.length);
            this.ctx.stroke();

            p.y += p.speed  * dt;
            p.x += p.wind   * dt;

            // Recycle drop by mutating in place — avoids Object.assign temp allocation
            if (p.y > this.H + 20 || p.x > this.W + 20) {
                p.x      = Math.random() * this.W;
                p.y      = -20;
                p.speed  = 400 + Math.random() * 300;
                p.length = 12  + Math.random() * 22;
                p.opacity= 0.15 + Math.random() * 0.45;
                p.wind   = 80  + Math.random() * 60;
            }
        }
        this.ctx.globalAlpha = 1;
    }
}


/* ====================================================================
   5.  STATE CONTROLLER — smooth lerp between named states
   ==================================================================== */
class StateController {
    constructor() {
        this.currentKey = 'SUNNY';
        this.targetKey  = 'SUNNY';
        this._progress  = 1;       // 0 = start of transition, 1 = done
        this._duration  = 2.2;     // seconds for a full state transition

        // Deep-clone current as the "live" blended state object
        this._live = this._deepClone(STATES.SUNNY);
    }

    _deepClone(s) {
        return {
            skyTop:      [...s.skyTop],
            skyMid:      [...s.skyMid],
            skyBot:      [...s.skyBot],
            sunColor:    [...s.sunColor],
            sunAlpha:    s.sunAlpha,
            sunY:        s.sunY,
            cloudAlpha:  [...s.cloudAlpha],
            cloudColor:  [...s.cloudColor],
            fogAlpha:    s.fogAlpha,
            starAlpha:   s.starAlpha,
            rainIntensity: s.rainIntensity
        };
    }

    requestState(key) {
        if (key === this.targetKey) return;
        // Snapshot current live state as the new "from" baseline
        this._from    = this._deepClone(this._live);
        this.currentKey = this.targetKey;   // previous target is now "current from"
        this.targetKey  = key;
        this._progress  = 0;
    }

    /** Call every frame — returns the live interpolated state */
    update(dt) {
        if (this._progress >= 1) return this._live;

        this._progress = Math.min(1, this._progress + dt / this._duration);
        const t = easeInOut(this._progress);

        const from = this._from;
        const to   = STATES[this.targetKey];
        const L    = this._live;

        // Mutate _live arrays in-place instead of allocating new arrays each frame
        lerpRGBInto(from.skyTop,     to.skyTop,     t, L.skyTop);
        lerpRGBInto(from.skyMid,     to.skyMid,     t, L.skyMid);  // new mid-stop
        lerpRGBInto(from.skyBot,     to.skyBot,     t, L.skyBot);
        lerpRGBInto(from.sunColor,   to.sunColor,   t, L.sunColor);
        lerpRGBInto(from.cloudColor, to.cloudColor, t, L.cloudColor);
        L.sunAlpha      = lerp(from.sunAlpha,      to.sunAlpha,      t);
        L.sunY          = lerp(from.sunY,          to.sunY,          t);  // smooth glow position
        L.fogAlpha      = lerp(from.fogAlpha,      to.fogAlpha,      t);
        L.starAlpha     = lerp(from.starAlpha,     to.starAlpha,     t);
        L.rainIntensity = lerp(from.rainIntensity, to.rainIntensity, t);
        for (let i = 0; i < 3; i++) {
            L.cloudAlpha[i] = lerp(from.cloudAlpha[i], to.cloudAlpha[i], t);
        }

        if (this._progress >= 1) {
            this.currentKey = this.targetKey;
        }

        return L;
    }

    get isRaining() {
        return this._live.rainIntensity > 0.02;
    }
}


/* ====================================================================
   6.  LIGHTNING ENGINE
   ==================================================================== */
class LightningEngine {
    constructor() {
        this.el = document.getElementById('lightning-overlay');
        this._nextFlash = this._randomDelay();
        this._active = false;
        this._countdown = 0;
    }

    _randomDelay() {
        // 4–14 seconds between flashes
        return 4 + Math.random() * 10;
    }

    update(dt, rainIntensity) {
        if (rainIntensity < 0.5 || !this.el) return;

        this._countdown -= dt;
        if (this._countdown <= 0) {
            this._countdown = this._nextFlash;
            this._nextFlash = this._randomDelay();
            this._triggerFlash();
        }
    }

    _triggerFlash() {
        if (!this.el) return;
        // Use double-rAF to re-trigger CSS animation without forcing
        // a synchronous layout recalculation (avoid offsetWidth reflow).
        this.el.classList.remove('flash');
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                this.el.classList.add('flash');
            });
        });
    }
}


/* ====================================================================
   7.  SCROLL → STATE MAPPING
   ==================================================================== */
function getScrollState() {
    const scrollable = document.body.scrollHeight - window.innerHeight;
    if (scrollable <= 0) return 'SUNNY';
    const pct = window.scrollY / scrollable;
    if (pct < 0.30) return 'SUNNY';
    if (pct < 0.58) return 'SUNSET';
    return 'NIGHT';
}


/* ====================================================================
   8.  MAIN WeatherSystem — orchestrates everything
   ==================================================================== */
class WeatherSystem {
    constructor() {
        this.skyCvs   = document.getElementById('weather-canvas');
        this.rainCvs  = document.getElementById('rain-canvas');

        if (!this.skyCvs) {
            console.warn('[WeatherSystem] #weather-canvas not found.');
            return;
        }

        this.sky       = new SkyRenderer(this.skyCvs);
        this.sky.initClouds();

        this.rain      = this.rainCvs ? new RainEngine(this.rainCvs) : null;
        this.mouse     = new MouseParallax();
        this.state     = new StateController();
        this.lightning = new LightningEngine();

        this._autoMode     = true;
        this._lastTime     = performance.now();
        this._elapsed      = 0;

        this._bindUI();
        this._bindScroll();
        requestAnimationFrame(t => this._loop(t));
    }

    _bindScroll() {
        let ticking = false;
        window.addEventListener('scroll', () => {
            if (this._autoMode && !ticking) {
                requestAnimationFrame(() => {
                    this.state.requestState(getScrollState());
                    ticking = false;
                });
                ticking = true;
            }
        }, { passive: true });
    }

    _bindUI() {
        document.querySelectorAll('.weather-btn').forEach(btn => {
            btn.addEventListener('click', e => {
                const s = e.currentTarget.dataset.state;
                document.querySelectorAll('.weather-btn').forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');

                if (s === 'AUTO') {
                    this._autoMode = true;
                    this.state.requestState(getScrollState());
                } else {
                    this._autoMode = false;
                    this.state.requestState(s);
                }
            });
        });
    }

    _loop(now) {
        if (document.hidden) {
            requestAnimationFrame(t => this._loop(t));
            return;
        }

        const dt = Math.min((now - this._lastTime) / 1000, 0.1); // cap at 100ms
        this._lastTime = now;
        this._elapsed += dt;

        // Update all subsystems
        this.mouse.update(dt);
        const liveState = this.state.update(dt);
        this.sky.updateClouds(dt);
        this.sky.draw(liveState, this._elapsed, this.mouse);

        if (this.rain) {
            this.rain.setIntensity(liveState.rainIntensity);
            this.rain.draw(dt);
        }

        this.lightning.update(dt, liveState.rainIntensity);

        requestAnimationFrame(t => this._loop(t));
    }
}

/* ====================================================================
   9.  BOOT
   ==================================================================== */
document.addEventListener('DOMContentLoaded', () => {
    window._weatherSystem = new WeatherSystem();
});
