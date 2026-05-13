import React, { useEffect, useRef } from 'react';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

// 组件 Props 类型
interface AvatarModelProps {
  url: string;                          // 模型文件的路径
  onHeadMeshReady?: (mesh: THREE.Mesh) => void;  // 回传头部 Mesh 供驱动组件使用
}

// 组件外预加载，当模块首次加载时即开始下载模型文件，提升性能
useGLTF.preload('/models/avatar.glb');

const AvatarModel: React.FC<AvatarModelProps> = ({ url, onHeadMeshReady }) => {
  // useGLTF 自动加载并缓存模型
  const gltf = useGLTF(url);
  const modelRef = useRef<THREE.Group>(null); // 引用整个模型组

  useEffect(() => {
    const meshesWithMorph: THREE.Mesh[] = [];

    // 遍历场景中所有子对象，找出包含 morphTargetDictionary 的 Mesh
    gltf.scene.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (
        mesh.isMesh &&                          // 确实是网格
        mesh.morphTargetDictionary &&           // 包含表情名称映射
        mesh.morphTargetInfluences              // 包含表情权重数组
      ) {
        meshesWithMorph.push(mesh);
      }
    });

    if (meshesWithMorph.length > 0) {
      // 通常 Ready Player Me 模型头部是第一个带 morph 的 Mesh
      const headMesh = meshesWithMorph[0];
      console.log('找到 morph 网格:', headMesh.name);
      console.log('可用表情目标:', Object.keys(headMesh.morphTargetDictionary));

      // 通过回调将头部 Mesh 传递给父组件
      if (onHeadMeshReady) {
        onHeadMeshReady(headMesh);
      }
    } else {
      // 如果没找到，说明模型可能没有 BlendShape，需要更换模型
      console.warn('没有找到任何含有 morphTargetDictionary 的 mesh！');
    }
  }, [gltf, onHeadMeshReady]); // 当模型加载完成或回调函数变化时执行

  // 把加载的模型添加到场景中
  return <primitive object={gltf.scene} ref={modelRef} />;
};

export default AvatarModel;