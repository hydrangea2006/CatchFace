// src/components/VrmModel.tsx
// 使用 @pixiv/three-vrm 的 VRMLoaderPlugin 正确加载 VRM 模型
// 替代原来的 Model.tsx + MorphWriter.tsx
import React, { useEffect, useRef, useMemo } from 'react';
import { useFrame, useLoader } from '@react-three/fiber';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';
import type { VRM } from '@pixiv/three-vrm';
import type { FaceControlState } from './OfflineFaceControl';

interface Props {
  url: string;
  onVrmReady?: (vrm: VRM) => void;
  onVrmDispose?: () => void;
  stateRef: React.RefObject<FaceControlState>;
}

/**
 * VrmModel 组件
 * - 使用 VRMLoaderPlugin 加载 .vrm 文件（也兼容 .glb 的 VRM 格式）
 * - 每帧从 stateRef 读取表情权重，通过 VRMExpressionManager 应用到模型
 * - 将 VRM 实例通过 onVrmReady 回调传给外部（用于动画控制等）
 */
const VrmModel: React.FC<Props> = ({ url, onVrmReady, onVrmDispose, stateRef }) => {
  // 使用 GLTFLoader + VRMLoaderPlugin 加载模型
  const gltf = useLoader(GLTFLoader, url, (loader) => {
    // 注册 VRM 加载器插件
    loader.register((parser) => {
      return new VRMLoaderPlugin(parser);
    });
  });

  // 从 gltf.userData 中取出 VRM 实例
  const vrm = useMemo(() => gltf.userData.vrm as VRM | undefined, [gltf]);

  // 保存当前已应用的表情权重，用于增量更新
  const appliedWeights = useRef<Map<string, number>>(new Map());

  // 通知父组件 VRM 已就绪
  const onVrmReadyRef = useRef(onVrmReady);
  onVrmReadyRef.current = onVrmReady;
  const onVrmDisposeRef = useRef(onVrmDispose);
  onVrmDisposeRef.current = onVrmDispose;

  useEffect(() => {
    if (vrm) {
      console.log('[VrmModel] VRM 模型加载成功');
      const allNames: string[] = Object.keys(vrm.expressionManager!.expressionMap!);
      console.log('[VrmModel] 可用表情:', allNames);

      // 将 gltf 内嵌的动画 clip 存到 VRM 实例上，供 AnimationController 使用
      if (gltf.animations && gltf.animations.length > 0) {
        (vrm as any)._gltfAnimations = gltf.animations;
        console.log('[VrmModel] 检测到内嵌动画:', gltf.animations.map((c) => c.name));
      }

      onVrmReadyRef.current?.(vrm);
    }

    return () => {
      onVrmDisposeRef.current?.();
      // 重置所有表情
      if (vrm?.expressionManager) {
        for (const name of Object.keys(vrm.expressionManager.expressionMap)) {
          vrm.expressionManager.setValue(name, 0);
        }
        vrm.expressionManager.update();
      }
    };
  }, [vrm]);

  // 每帧同步表情权重
  useFrame(() => {
    if (!vrm?.expressionManager) return;

    const em = vrm.expressionManager;
    const { weights } = stateRef.current;
    let needUpdate = false;

    // 处理当前设置的权重
    const activeNames = new Set<string>();

    for (const [name, value] of Object.entries(weights)) {
      activeNames.add(name);
      const finalValue = Math.min(1, Math.max(0, value));

      const prev = appliedWeights.current.get(name);
      if (prev !== finalValue) {
        em.setValue(name, finalValue);
        appliedWeights.current.set(name, finalValue);
        needUpdate = true;
      }
    }

    // 归零那些之前设置过但现在不在 weights 中的表情
    for (const [name] of appliedWeights.current) {
      if (!activeNames.has(name)) {
        em.setValue(name, 0);
        appliedWeights.current.delete(name);
        needUpdate = true;
      }
    }

    if (needUpdate) {
      em.update();
    }
  });

  // 如果 VRM 加载成功，渲染其 scene
  if (vrm) {
    return <primitive object={vrm.scene} />;
  }

  // 否则渲染原始的 gltf scene（理论上不会走到这里，因为 vrm 总是有值）
  return <primitive object={gltf.scene} />;
};

export default VrmModel;
