// src/components/VrmModel.tsx
// VRM 模型加载 + ARkit blendshape 表情驱动 + 头部姿态同步
//
// 驱动策略（按优先级自动选择）：
//   A) expressionManager.setValue()
//   B) mesh.morphTargetInfluences[] — 直接操作 morph target 权重
//   C) bone rotation — 骨骼旋转驱动
import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, type VRM } from '@pixiv/three-vrm';
import { computeExpressions } from '../utils/arkitsToExpression';
import * as THREE from 'three';

export interface HeadPose {
  rotation: { x: number; y: number; z: number };
  position: number[];
}

export interface ArkitsFrame {
  timestamp: number;
  head: HeadPose;
  blendshapes: Record<string, number>;
}

interface Props {
  url: string;
  onVrmReady?: (vrm: VRM) => void;
  frameRef: React.RefObject<ArkitsFrame | null>;
}

interface MorphMapping {
  mesh: THREE.Mesh;
  nameToIndex: Map<string, number>;
}

// 骨骼驱动映射配置（Vroid 常见命名）
interface BoneDriver {
  boneName: string;
  axis: 'x' | 'y' | 'z';
  maxAngle: number;
  defaultValue?: number;
}

const BONE_EXPRESSION_MAP: Record<string, BoneDriver[]> = {
  eyeBlinkLeft: [
    { boneName: 'Facial_eye_L', axis: 'z', maxAngle: -0.6 },
    { boneName: 'Face_eye_L', axis: 'z', maxAngle: -0.6 },
    { boneName: 'eye_L', axis: 'z', maxAngle: -0.6 },
  ],
  eyeBlinkRight: [
    { boneName: 'Facial_eye_R', axis: 'z', maxAngle: -0.6 },
    { boneName: 'Face_eye_R', axis: 'z', maxAngle: -0.6 },
    { boneName: 'eye_R', axis: 'z', maxAngle: -0.6 },
  ],
  jawOpen: [
    { boneName: 'Jaw', axis: 'x', maxAngle: 0.6 },
    { boneName: 'Face_jaw', axis: 'x', maxAngle: 0.6 },
    { boneName: 'Facial_jaw', axis: 'x', maxAngle: 0.6 },
  ],
  mouthSmileLeft: [
    { boneName: 'Facial_mouth_L', axis: 'y', maxAngle: 0.3 },
    { boneName: 'Face_mouth_L', axis: 'y', maxAngle: 0.3 },
    { boneName: 'Facial_mouth_corner_L', axis: 'y', maxAngle: 0.3 },
  ],
  mouthSmileRight: [
    { boneName: 'Facial_mouth_R', axis: 'y', maxAngle: -0.3 },
    { boneName: 'Face_mouth_R', axis: 'y', maxAngle: -0.3 },
    { boneName: 'Facial_mouth_corner_R', axis: 'y', maxAngle: -0.3 },
  ],
  browInnerUp: [
    { boneName: 'Facial_brow_inner_L', axis: 'x', maxAngle: -0.3 },
    { boneName: 'Facial_brow_inner_R', axis: 'x', maxAngle: -0.3 },
    { boneName: 'Face_brow_inner_L', axis: 'x', maxAngle: -0.3 },
    { boneName: 'Face_brow_inner_R', axis: 'x', maxAngle: -0.3 },
  ],
  browOuterUpLeft: [
    { boneName: 'Facial_brow_outer_L', axis: 'x', maxAngle: -0.3 },
    { boneName: 'Face_brow_outer_L', axis: 'x', maxAngle: -0.3 },
  ],
  browOuterUpRight: [
    { boneName: 'Facial_brow_outer_R', axis: 'x', maxAngle: -0.3 },
    { boneName: 'Face_brow_outer_R', axis: 'x', maxAngle: -0.3 },
  ],
  mouthPucker: [
    { boneName: 'Facial_mouth_L', axis: 'x', maxAngle: 0.2 },
    { boneName: 'Facial_mouth_R', axis: 'x', maxAngle: -0.2 },
    { boneName: 'Face_mouth_L', axis: 'x', maxAngle: 0.2 },
    { boneName: 'Face_mouth_R', axis: 'x', maxAngle: -0.2 },
  ],
  mouthFunnel: [
    { boneName: 'Facial_mouth_L', axis: 'x', maxAngle: 0.25 },
    { boneName: 'Facial_mouth_R', axis: 'x', maxAngle: -0.25 },
    { boneName: 'Face_mouth_L', axis: 'x', maxAngle: 0.25 },
    { boneName: 'Face_mouth_R', axis: 'x', maxAngle: -0.25 },
  ],
  eyeWideLeft: [
    { boneName: 'Facial_eye_L', axis: 'z', maxAngle: 0.4 },
    { boneName: 'Face_eye_L', axis: 'z', maxAngle: 0.4 },
  ],
  eyeWideRight: [
    { boneName: 'Facial_eye_R', axis: 'z', maxAngle: 0.4 },
    { boneName: 'Face_eye_R', axis: 'z', maxAngle: 0.4 },
  ],
};

const VrmModel: React.FC<Props> = ({ url, onVrmReady, frameRef }) => {
  const gltf = useLoader(GLTFLoader, url, (loader) => {
    loader.register((parser) => new VRMLoaderPlugin(parser));
  });

  const vrm = useMemo(() => gltf.userData.vrm as VRM | undefined, [gltf]);

  const useExpressionManager = useRef(false);
  const registeredExpressions = useRef<Set<string>>(new Set());
  const morphMappings = useRef<MorphMapping[]>([]);
  const boneDrivers = useRef<Map<string, Array<{ node: THREE.Object3D; axis: string; maxAngle: number; defaultValue: number }>>>(new Map());
  const debugFrameCount = useRef(0);

  useEffect(() => {
    if (!vrm) return;

    console.log('[VrmModel] ✅ VRM 加载成功');

    // ── 方案 A：expressionManager ──
    const em = vrm.expressionManager;
    if (em) {
      // 兼容不同版本的 VRM SDK：尝试获取已注册的表达式名称
      let names: string[] = [];
      try {
        // v1.x: expressionsMap 是 Map<string, VRMExpression>
        const exprMap = (em as any).expressionsMap;
        if (exprMap instanceof Map) {
          names = [...exprMap.keys()];
        }
      } catch {
        // 忽略
      }
      if (names.length > 0) {
        useExpressionManager.current = true;
        registeredExpressions.current = new Set(names);
        console.log('[VrmModel] 📋 使用 ExpressionManager (' + names.length + '个表达式):', names);
        onVrmReady?.(vrm);
        return;
      } else {
        console.log('[VrmModel] 📋 ExpressionManager 无已注册表达式，fallback 到 morph target');
      }
    }

    // ── 方案 B：morph targets ──
    const mappings: MorphMapping[] = [];
    const allMorphNames = new Set<string>();

    vrm.scene.traverse((obj) => {
      if ((obj as THREE.Mesh).isMesh) {
        const mesh = obj as THREE.Mesh;
        const dict = mesh.morphTargetDictionary;
        const influences = mesh.morphTargetInfluences;
        if (dict && influences && influences.length > 0) {
          const nameToIndex = new Map<string, number>();
          for (const [name, index] of Object.entries(dict)) {
            nameToIndex.set(name, Number(index));
            allMorphNames.add(name);
          }
          mappings.push({ mesh, nameToIndex });
        }
      }
    });

    morphMappings.current = mappings;

    if (mappings.length > 0) {
      console.log('[VrmModel] 🎭 Morph targets: ' + allMorphNames.size + '个');
      console.log('[VrmModel] 📋 使用 Morph target 直驱模式');
      console.log('[VrmModel] 🎭 关键 ARkit 名匹配检查:',
        'eyeBlinkLeft=' + allMorphNames.has('eyeBlinkLeft'),
        'jawOpen=' + allMorphNames.has('jawOpen'),
        'mouthSmileLeft=' + allMorphNames.has('mouthSmileLeft'),
        'browInnerUp=' + allMorphNames.has('browInnerUp'),
        'mouthPucker=' + allMorphNames.has('mouthPucker'),
      );
      // 打印所有 ShapeKey 名，确认 facial1. 前缀
      console.log('[VrmModel] 🎭 全部 ShapeKey 名称:', [...allMorphNames].sort());
      // 不 return，继续执行下面的手臂下垂逻辑
    } else {
      // ── 方案 C：骨骼驱动 ──
    console.log('[VrmModel] 🔍 启用骨骼驱动模式');

    const boneNodeMap = new Map<string, THREE.Object3D>();
    vrm.scene.traverse((obj) => {
      if (obj.name) boneNodeMap.set(obj.name, obj);
    });

    const faceRelatedNames = [...boneNodeMap.keys()].filter(
      (n) => /face|jaw|eye|brow|mouth|cheek|nose|head|neck/i.test(n)
    );
    console.log('[VrmModel] 🦴 面部相关骨骼节点 (' + faceRelatedNames.length + '个):', faceRelatedNames.sort());

    const drivers = new Map<string, Array<{ node: THREE.Object3D; axis: string; maxAngle: number; defaultValue: number }>>();

    for (const [arkitName, configs] of Object.entries(BONE_EXPRESSION_MAP)) {
      const matched: Array<{ node: THREE.Object3D; axis: string; maxAngle: number; defaultValue: number }> = [];
      for (const config of configs) {
        const node = boneNodeMap.get(config.boneName);
        if (node) {
          matched.push({ node, axis: config.axis, maxAngle: config.maxAngle, defaultValue: node.rotation[config.axis] });
        }
      }
      if (matched.length > 0) drivers.set(arkitName, matched);
    }

    boneDrivers.current = drivers;
    console.log('[VrmModel] 🎬 骨骼驱动映射 (' + drivers.size + '个表达式):', [...drivers.keys()]);

    if (drivers.size === 0) {
      console.warn('[VrmModel] ❌ 未能匹配任何面部骨骼！');
      console.warn('[VrmModel] 查找的骨骼名:',
        [...new Set(Object.values(BONE_EXPRESSION_MAP).flat().map(c => c.boneName))].sort()
      );
      console.warn('[VrmModel] 模型节点:', [...boneNodeMap.keys()].sort());
    }
    } // end else (方案 C: 骨骼驱动)

    onVrmReady?.(vrm);
  }, [vrm, onVrmReady]);

  // ── 每帧驱动 ──
  useFrame((_state, delta) => {
    if (!vrm) return;

    const frame = frameRef.current;

    if (frame) {
      const { blendshapes } = frame;

      // ─── 方案 A：expressionManager ───
      if (useExpressionManager.current) {
        const em = vrm.expressionManager;
        if (em) {
          for (const [name, value] of Object.entries(blendshapes)) {
            if (registeredExpressions.current.has(name)) {
              em.setValue(name, Math.min(1, Math.max(0, value)));
            }
          }
          em.update();
        }
      }

      // ─── 方案 C：骨骼驱动 ───
      if (boneDrivers.current.size > 0) {
        for (const [arkitName, value] of Object.entries(blendshapes)) {
          const drivers = boneDrivers.current.get(arkitName);
          if (drivers) {
            const clamped = Math.min(1, Math.max(0, value));
            for (const d of drivers) {
              const angle = d.defaultValue + d.maxAngle * clamped;
              (d.node.rotation as any)[d.axis] = angle;
            }
          }
        }
      }

      // ─── 头部旋转 ───
      const { rotation } = frame.head;
      const humanoid = vrm.humanoid;
      if (humanoid?.humanBones) {
        const headBone = humanoid.humanBones.head?.node;
        const neckBone = humanoid.humanBones.neck?.node;
        if (headBone) {
          headBone.rotation.set(rotation.x, rotation.y, rotation.z);
        }
        if (neckBone) {
          neckBone.rotation.set(rotation.x * 0.5, rotation.y * 0.5, rotation.z * 0.3);
        }
        humanoid.update();
      }
    }


    debugFrameCount.current++;

    const hb = vrm.humanoid?.humanBones;

    // ⚠️ 调用 vrm.update(delta)，内部处理 expressionManager + humanoid
    vrm.update(delta);

    // ─── 方案 B：morph target 直驱 —— 必须在 vrm.update() 之后执行 ───
    // 因为 vrm.update() 内部可能重置 morphTargetInfluences 数组
    if (frame && morphMappings.current.length > 0 && !useExpressionManager.current) {
      const { blendshapes } = frame;

      // 计算衍生表情（ARkit → VRM ShapeKey 映射）
      const expressions = computeExpressions(blendshapes);

      // 调试：有衍生表情时打印
      const exprKeys = Object.keys(expressions);
      if (exprKeys.length > 0 && debugFrameCount.current % 60 === 1) {
        console.log('[VrmModel] 🎭 衍生表情:', Object.entries(expressions).map(([k, v]) => `${k}=${v.toFixed(2)}`));
      }

      // 合并：原始 ARkit 值 + 衍生表情值（衍生表情优先级更高，后写入覆盖）
      const allValues: Record<string, number> = { ...blendshapes, ...expressions };

      for (const mapping of morphMappings.current) {
        const arr = mapping.mesh.morphTargetInfluences!;
        for (const [key, value] of Object.entries(allValues)) {
          const idx = mapping.nameToIndex.get(key);
          if (idx !== undefined && idx < arr.length) {
            arr[idx] = value;
          }
        }
      }

      // 调试：每60帧打印一次关键 morph 值
      debugFrameCount.current++;
      if (debugFrameCount.current % 60 === 0) {
        const m = morphMappings.current[0];
        if (m) {
          const eyeIdx = m.nameToIndex.get('eyeBlinkLeft');
          const jawIdx = m.nameToIndex.get('jawOpen');
          const smileIdx = m.nameToIndex.get('mouthSmileLeft');
          console.log(
            `[VrmModel] 🔍 帧#${debugFrameCount.current} (vrm.update后)`,
            `eyeBlinkLeft=${eyeIdx !== undefined ? m.mesh.morphTargetInfluences![eyeIdx].toFixed(3) : 'N/A'}`,
            `jawOpen=${jawIdx !== undefined ? m.mesh.morphTargetInfluences![jawIdx].toFixed(3) : 'N/A'}`,
            `mouthSmileLeft=${smileIdx !== undefined ? m.mesh.morphTargetInfluences![smileIdx].toFixed(3) : 'N/A'}`,
          );
        }
      }
    }
  });

  if (vrm) {
    return <primitive object={vrm.scene} />;
  }
  return <primitive object={gltf.scene} />;
};

export default VrmModel;
