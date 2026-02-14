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
      void main() {
        vec4 worldPos = modelMatrix * vec4(position, 1.0);
        vWorldPos = worldPos.xyz;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        vUv = uv;
        gl_Position = projectionMatrix * viewMatrix * worldPos;
      }
    `,
    fragmentShader: `
      precision highp float;
      varying vec3 vWorldPos;
      varying vec3 vNormalW;
      varying vec2 vUv;
      uniform vec3 uBaseColor;
      uniform sampler2D uBaseMap;
      uniform float uHasMap;
      uniform float uSeverity;
      uniform float uTime;

      float hash31(vec3 p) {
        p = fract(p * 0.1031);
        p += dot(p, p.yzx + 33.33);
        return fract((p.x + p.y) * p.z);
      }

      float noise3(vec3 p) {
        vec3 i = floor(p);
        vec3 f = fract(p);
        f = f * f * (3.0 - 2.0 * f);

        float n000 = hash31(i + vec3(0.0, 0.0, 0.0));
        float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
        float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
        float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
        float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
        float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
        float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
        float n111 = hash31(i + vec3(1.0, 1.0, 1.0));

        float nx00 = mix(n000, n100, f.x);
        float nx10 = mix(n010, n110, f.x);
        float nx01 = mix(n001, n101, f.x);
        float nx11 = mix(n011, n111, f.x);

        float nxy0 = mix(nx00, nx10, f.y);
        float nxy1 = mix(nx01, nx11, f.y);

        return mix(nxy0, nxy1, f.z);
      }

      float fbmLow(vec3 p) {
        float v = 0.0;
        float a = 0.55;
        for (int i = 0; i < 3; i++) {
          v += a * noise3(p);
          p *= 1.9;
          a *= 0.55;
        }
        return v;
      }

      vec3 severityColor(float s) {
        if (s >= 0.7) return vec3(0.8980, 0.2235, 0.2078);
        if (s >= 0.4) return vec3(1.0, 0.5961, 0.0);
        return vec3(1.0, 0.8353, 0.3098);
      }

      void main() {
        float sev = clamp(uSeverity, 0.0, 1.0);
        vec3 baseCol = uBaseColor;
        if (uHasMap > 0.5) {
          baseCol *= texture2D(uBaseMap, vUv).rgb;
        }

        vec3 lightDir = normalize(vec3(0.35, 0.85, 0.55));
        float ndl = clamp(dot(normalize(vNormalW), lightDir), 0.0, 1.0);
        float shade = 0.55 + 0.45 * ndl;

        if (sev < 0.2) {
          gl_FragColor = vec4(baseCol * shade, 1.0);
          return;
        }

        vec3 infection = severityColor(sev);

        float s = clamp((sev - 0.2) / 0.8, 0.0, 1.0);
        vec3 p = vWorldPos * 0.22 + vec3(0.0, 0.0, uTime * 0.025);
        float n = fbmLow(p);

        float coverage = mix(0.86, 0.62, s);
        float softness = mix(0.18, 0.22, s);
        float mask = smoothstep(coverage, coverage + softness, n);
        mask = pow(mask, 1.2);
        mask *= (0.12 + 0.88 * s);
        mask = min(mask, 0.85);

        vec3 mixed = mix(baseCol, infection, mask);
        gl_FragColor = vec4(mixed * shade, 1.0);
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
      <ambientLight intensity={0.7} />
      <directionalLight intensity={0.8} position={[2.5, 3.5, 4]} />
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
