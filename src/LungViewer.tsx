import * as THREE from 'three';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import { clamp01 } from './severity';

type LungModelProps = {
  url: string;
  targetSeverity: number;
  animate: boolean;
};

function createLungShaderMaterial(params: {
  baseColor: THREE.Color;
  baseMap: THREE.Texture | null;
  side: THREE.Side;
}): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uBaseColor: { value: params.baseColor },
      uBaseMap: { value: params.baseMap },
      uHasMap: { value: params.baseMap ? 1.0 : 0.0 },
      uSeverity: { value: 0.0 },
      uTime: { value: 0.0 }
    },
    vertexShader: `
      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying vec2 vUv;
      varying vec3 vViewDir;
      void main() {
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPos.xyz;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        vUv = uv;
        vViewDir = normalize(cameraPosition - worldPos.xyz);
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: `
      precision highp float;
      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying vec2 vUv;
      varying vec3 vViewDir;
      uniform vec3 uBaseColor;
      uniform sampler2D uBaseMap;
      uniform float uHasMap;
      uniform float uSeverity;
      uniform float uTime;

      /* ---- Hash & Noise ---- */
      float hash31(vec3 p) {
        p = fract(p * 0.1031);
        p += dot(p, p.yzx + 33.33);
        return fract((p.x + p.y) * p.z);
      }

      float noise3(vec3 p) {
        vec3 i = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);
        float n000 = hash31(i);
        float n100 = hash31(i + vec3(1,0,0));
        float n010 = hash31(i + vec3(0,1,0));
        float n110 = hash31(i + vec3(1,1,0));
        float n001 = hash31(i + vec3(0,0,1));
        float n101 = hash31(i + vec3(1,0,1));
        float n011 = hash31(i + vec3(0,1,1));
        float n111 = hash31(i + vec3(1,1,1));
        return mix(mix(mix(n000,n100,f.x), mix(n010,n110,f.x), f.y),
                   mix(mix(n001,n101,f.x), mix(n011,n111,f.x), f.y), f.z);
      }

      /* Multi-octave FBM for clustered organic patches */
      float fbm(vec3 p) {
        float v = 0.0, a = 0.5;
        for (int i = 0; i < 5; i++) {
          v += a * noise3(p);
          p *= 2.1;
          a *= 0.48;
        }
        return v;
      }

      /* Secondary detail noise for infection texture */
      float detailNoise(vec3 p) {
        return noise3(p * 3.5) * 0.5 + noise3(p * 7.0) * 0.25 + noise3(p * 14.0) * 0.125;
      }

      /* ---- Healthy tissue tint by severity level ---- */
      vec3 healthyTint(float sev) {
        // Low: cool blue-teal (healthy), Mild: muted green-blue,
        // Moderate: pale amber, Severe: desaturated/stressed tissue
        vec3 low      = vec3(0.55, 0.82, 0.88);  // teal
        vec3 mild     = vec3(0.60, 0.78, 0.65);  // sage green
        vec3 moderate = vec3(0.85, 0.75, 0.50);  // amber
        vec3 severe   = vec3(0.68, 0.55, 0.55);  // stressed pink-gray

        if (sev < 0.2)  return low;
        if (sev < 0.4)  return mix(low, mild, (sev - 0.0) / 0.4);
        if (sev < 0.7)  return mix(mild, moderate, (sev - 0.4) / 0.3);
        return mix(moderate, severe, (sev - 0.7) / 0.3);
      }

      /* ---- Infection hot-spot color ---- */
      vec3 infectionCore(float sev, float depth) {
        // Core of infection: bright, high-saturation
        vec3 mildInf     = vec3(1.0, 0.85, 0.2);   // warm yellow
        vec3 moderateInf = vec3(1.0, 0.45, 0.1);   // hot orange
        vec3 severeInf   = vec3(0.95, 0.15, 0.12);  // bright red

        vec3 col;
        if (sev < 0.4) col = mix(mildInf, moderateInf, sev / 0.4);
        else           col = mix(moderateInf, severeInf, (sev - 0.4) / 0.6);

        // Brighter at center, darker at edges of patches
        col += depth * vec3(0.15, 0.08, 0.02);
        return col;
      }

      void main() {
        float sev = clamp(uSeverity, 0.0, 1.0);
        vec3 baseCol = uBaseColor;
        if (uHasMap > 0.5) {
          baseCol *= texture2D(uBaseMap, vUv).rgb;
        }

        /* -- Lighting -- */
        vec3 N = normalize(vNormalW);
        vec3 V = normalize(vViewDir);
        vec3 L = normalize(vec3(0.35, 0.85, 0.55));
        float ndl = max(dot(N, L), 0.0);
        float shade = 0.50 + 0.50 * ndl;

        // Fresnel rim light for depth perception
        float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);

        /* -- Healthy tissue base color by severity -- */
        vec3 healthyCol = healthyTint(sev) * baseCol;
        // Add subtle rim highlight on healthy areas
        healthyCol += fresnel * 0.12 * vec3(0.6, 0.8, 1.0);

        /* -- If very low severity, show mostly healthy -- */
        if (sev < 0.05) {
          gl_FragColor = vec4(healthyCol * shade, 1.0);
          return;
        }

        /* -- Compute infection mask (clustered patches) -- */
        float s = clamp(sev, 0.0, 1.0);

        // Primary patch noise — slow drift for organic feel
        vec3 noiseCoord = vWorldPos * 0.28 + vec3(0.0, 0.0, uTime * 0.012);
        float n = fbm(noiseCoord);

        // Coverage grows with severity: tiny spots at low sev → large areas at high
        float coverage = mix(0.82, 0.42, s);    // threshold lowers = more area
        float edgeWidth = mix(0.06, 0.10, s);   // sharper edges than before

        // Primary infection mask
        float mask = smoothstep(coverage, coverage + edgeWidth, n);

        // At low severity, reduce overall intensity so only hints appear
        float intensityScale = smoothstep(0.0, 0.25, s);
        mask *= intensityScale;

        // Detail variation inside infected patches (texture/depth)
        float detail = detailNoise(vWorldPos * 0.6 + uTime * 0.005);
        float innerDepth = mask * detail;

        /* -- Edge glow at infection boundaries -- */
        float edgeMask = smoothstep(coverage - 0.03, coverage, n)
                       - smoothstep(coverage, coverage + edgeWidth * 0.5, n);
        edgeMask = max(edgeMask, 0.0) * intensityScale;

        // Pulsing glow
        float pulse = 0.8 + 0.2 * sin(uTime * 1.8 + n * 6.0);
        edgeMask *= pulse;

        /* -- Infection color with depth variation -- */
        vec3 infColor = infectionCore(sev, innerDepth);

        // Emissive boost inside infection (self-illuminated look)
        float emissive = mask * mix(0.15, 0.45, s) * pulse;

        /* -- Edge highlight (bright rim around infected patches) -- */
        vec3 edgeGlow = vec3(1.0, 0.7, 0.2) * edgeMask * 1.5;
        if (sev >= 0.7) edgeGlow = vec3(1.0, 0.3, 0.15) * edgeMask * 1.8;

        /* -- Composite: blend healthy + infection -- */
        vec3 finalColor = mix(healthyCol, infColor, mask);

        // Apply lighting
        finalColor *= shade;

        // Add emissive glow to infected areas (not affected by shade)
        finalColor += infColor * emissive;

        // Add edge glow
        finalColor += edgeGlow;

        // Subtle fresnel on infected areas for dramatic effect
        finalColor += fresnel * mask * 0.15 * infColor;

        gl_FragColor = vec4(finalColor, 1.0);
      }
    `,
    transparent: false,
    depthWrite: true,
    depthTest: true,
    side: params.side
  });
}

function LungModel({ url, targetSeverity, animate }: LungModelProps) {
  const gltf = useGLTF(url);
  const { camera } = useThree();
  const [materials, setMaterials] = useState<THREE.ShaderMaterial[]>([]);
  const currentSeverity = useRef(0);

  const scene = useMemo(() => {
    const cloned = gltf.scene.clone(true);
    return cloned;
  }, [gltf.scene]);

  useEffect(() => {
    const mats: THREE.ShaderMaterial[] = [];

    scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;

      const origMat = mesh.material as THREE.MeshStandardMaterial | THREE.MeshPhysicalMaterial | THREE.Material;
      const color = (origMat as any).color instanceof THREE.Color ? ((origMat as any).color as THREE.Color).clone() : new THREE.Color(0.82, 0.82, 0.82);
      const map = (origMat as any).map instanceof THREE.Texture ? ((origMat as any).map as THREE.Texture) : null;
      const side = typeof (origMat as any).side === 'number' ? ((origMat as any).side as THREE.Side) : THREE.FrontSide;

      const shaderMat = createLungShaderMaterial({ baseColor: color, baseMap: map, side });
      mesh.material = shaderMat;
      mats.push(shaderMat);
    });

    if (mats.length === 0) {
      console.error('GLB loaded but contained no mesh surfaces. Rendering nothing.');
    }

    setMaterials(mats);

    const box = new THREE.Box3().setFromObject(scene);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);

    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    camera.near = Math.max(0.01, maxDim / 1000);
    camera.far = maxDim * 50;
    camera.position.set(center.x, center.y, center.z + maxDim * 2.2);
    camera.lookAt(center);
    camera.updateProjectionMatrix();
  }, [camera, scene]);

  useFrame((state, delta) => {
    const t = state.clock.getElapsedTime();
    const target = clamp01(targetSeverity);

    if (animate) {
      currentSeverity.current = THREE.MathUtils.lerp(currentSeverity.current, target, 1.0 - Math.pow(0.0005, delta));
    } else {
      currentSeverity.current = target;
    }

    for (const m of materials) {
      m.uniforms.uSeverity.value = currentSeverity.current;
      m.uniforms.uTime.value = t;
    }
  });

  if (!scene) return null;
  return <primitive object={scene} />;
}

export default function LungViewer(props: {
  severity: number;
  animate: boolean;
}) {
  return (
    <Canvas
      camera={{ fov: 45, near: 0.01, far: 1000 }}
      gl={{ antialias: true, alpha: false }}
      style={{ width: '100%', height: '100%', background: '#0b0b10' }}
    >
      <ambientLight intensity={0.55} />
      <directionalLight intensity={0.9} position={[2.5, 3.5, 4]} />
      <directionalLight intensity={0.3} position={[-3, 1, -2]} color="#aaccff" />
      <OrbitControls enableDamping dampingFactor={0.08} />
      <LungModel
        url={'/public/models/lung_carcinoma.glb'}
        targetSeverity={props.severity}
        animate={props.animate}
      />
    </Canvas>
  );
}

useGLTF.preload('/public/models/lung_carcinoma.glb');
