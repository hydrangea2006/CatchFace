// src/components/AnimationController.tsx
// VRM 动画控制器
// - 自动检测 VRM 模型内嵌的 glTF 动画
// - 支持手动加载外部 .vrma 动画文件
import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation';
import type { VRMAnimation } from '@pixiv/three-vrm-animation';

interface Props {
  vrm: VRM | null;
}

interface LoadedAnim {
  name: string;
  action: THREE.AnimationAction;
}

const AnimationController: React.FC<Props> = ({ vrm }) => {
  const [animations, setAnimations] = useState<LoadedAnim[]>([]);
  const [playing, setPlaying] = useState<string | null>(null);
  const mixerRef = useRef<THREE.AnimationMixer | null>(null);
  const currentActionRef = useRef<THREE.AnimationAction | null>(null);

  // vrm 变化时：检测内嵌动画 + 清理旧动画
  useEffect(() => {
    setAnimations([]);
    setPlaying(null);
    currentActionRef.current = null;
    mixerRef.current = null;

    if (!vrm) return;

    // 检查 VRM 的 gltf 是否带内嵌动画
    const gltfAnimations = (vrm as any)._gltfAnimations as THREE.AnimationClip[] | undefined;
    if (gltfAnimations && gltfAnimations.length > 0) {
      const mixer = new THREE.AnimationMixer(vrm.scene);
      mixerRef.current = mixer;
      const loaded: LoadedAnim[] = gltfAnimations.map((clip) => {
        const action = mixer.clipAction(clip);
        action.setLoop(THREE.LoopRepeat, Infinity);
        return { name: clip.name || '未命名动画', action };
      });
      setAnimations(loaded);
      console.log('[AnimationController] 检测到内嵌动画:', loaded.map((a) => a.name));
    }
  }, [vrm]);

  // 每帧更新 mixer + 同步 VRM humanoid
  useEffect(() => {
    let rafId: number;
    const clock = new THREE.Clock();
    const loop = () => {
      rafId = requestAnimationFrame(loop);
      const delta = Math.min(clock.getDelta(), 0.1); // 防止大 delta 导致跳帧
      if (mixerRef.current) {
        mixerRef.current.update(delta);
        // 关键：AnimationMixer 修改的是 normalized bones，需要调用 humanoid.update()
        // 将 normalized bones 的姿势同步到 raw bones（实际渲染的骨骼）
        vrm?.humanoid?.update();
      }
    };
    loop();
    return () => cancelAnimationFrame(rafId);
  }, [vrm]);

  // 从本地文件加载 .vrma 动画
  const loadFromFile = useCallback(
    (file: File) => {
      if (!vrm) return;

      const name = file.name.replace(/\.vrma$/i, '');
      const url = URL.createObjectURL(file);

      const loader = new GLTFLoader();
      loader.register((parser) => new VRMAnimationLoaderPlugin(parser));

      loader.load(
        url,
        (gltf) => {
          URL.revokeObjectURL(url);
          const vrmAnimations = (gltf as any).userData.vrmAnimations as VRMAnimation[] | undefined;
          if (!vrmAnimations || vrmAnimations.length === 0) {
            alert('未能解析 VRM 动画文件');
            return;
          }
          const vrmAnimation = vrmAnimations[0];

          if (!mixerRef.current) {
            mixerRef.current = new THREE.AnimationMixer(vrm.scene);
          }

          // 使用官方 createVRMAnimationClip 创建动画剪辑
          const clip = createVRMAnimationClip(vrmAnimation, vrm);

          clip.name = name;

          const action = mixerRef.current.clipAction(clip);
          action.setLoop(THREE.LoopRepeat, Infinity);

          setAnimations((prev) => [...prev, { name, action }]);
          console.log(`[AnimationController] 加载动画成功: ${name}, tracks: ${clip.tracks.length}`);
        },
        undefined,
        (err) => {
          URL.revokeObjectURL(url);
          console.error('[AnimationController] 动画加载失败:', err);
          alert(`动画加载失败: ${file.name}`);
        },
      );
    },
    [vrm],
  );

  // 播放
  const playAnimation = useCallback((anim: LoadedAnim) => {
    if (currentActionRef.current) {
      currentActionRef.current.stop();
    }
    anim.action.reset().play();
    currentActionRef.current = anim.action;
    setPlaying(anim.name);
  }, []);

  // 停止
  const stopAnimation = useCallback(() => {
    currentActionRef.current?.stop();
    currentActionRef.current = null;
    setPlaying(null);
  }, []);

  if (!vrm) return null;

  return (
    <div
      style={{
        position: 'absolute',
        top: 10,
        right: 10,
        background: '#ffffffcc',
        padding: 12,
        borderRadius: 8,
        fontFamily: 'sans-serif',
        minWidth: 200,
      }}
    >
      <h3 style={{ marginTop: 0 }}>动画控制</h3>

      {animations.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <small>已加载动画：</small>
          {animations.map((anim) => (
            <div key={anim.name} style={{ marginBottom: 4 }}>
              <strong>{anim.name}</strong>
              {playing === anim.name && (
                <span style={{ color: '#4a90e2', fontSize: '0.8em', marginLeft: 8 }}>
                  ▶ 播放中
                </span>
              )}
              <div>
                <button
                  onClick={() => playAnimation(anim)}
                  style={{ padding: '2px 8px', cursor: 'pointer', fontSize: '0.8em' }}
                >
                  ▶ 播放
                </button>
                <button
                  onClick={stopAnimation}
                  style={{ padding: '2px 8px', cursor: 'pointer', fontSize: '0.8em' }}
                >
                  ⏹ 停止
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div>
        <small>从本地加载 .vrma 动画：</small>
        <input
          type="file"
          accept=".vrma"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) loadFromFile(file);
          }}
          style={{ marginTop: 4, fontSize: '0.8em' }}
        />
      </div>
    </div>
  );
};

export default AnimationController;
