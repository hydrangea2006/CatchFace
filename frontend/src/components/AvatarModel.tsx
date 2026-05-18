import React, { useEffect, useRef } from 'react';
// 加载3D模型的自定义Hook
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

interface AvatarModelProps {
  url: string;  //模型文件的地址
  onHeadMeshReady?: (mesh: THREE.Mesh) => void; //回调函数，将找到的网络交给父组件驱动
}

// 提前下载模型文件
useGLTF.preload('/models/avatar.glb');

// React.FC为React函数组件，自带children类型
const AvatarModel: React.FC<AvatarModelProps> = ({url, onHeadMeshReady}) => {
  // 加载并缓存GLB模型
  const gltf = useGLTF(url);
  const modelRef = useRef<THREE.Group>(null);

  useEffect(() => {
    // 存放表情模型的数组
    const meshesWithMorph: THREE.Mesh[] = [];

    // 遍历模型的子元素，在其中找到带有表情数据的网格
    gltf.scene.traverse((child) => {
      const mesh = child as THREE.Mesh;
      // 如果mesh是3D网格，有表情列表，并且有表情强度控制数组，那么就添加到数组meshesWithMroph数组中

      //  //后端对接：映射表中表情数据有类似{"mouthOpen": 0,"mouthSmile": 1}的映射即可

        if (mesh.isMesh && mesh.morphTargetDictionary && mesh.morphTargetInfluences) {
          meshesWithMorph.push(mesh);
        }     
    });

    if (meshesWithMorph.length > 0) {
      // 第一个网格为头部
      const headMesh = meshesWithMorph[0];
      console.log("找到可驱动表情的网格", headMesh.name);
      console.log("支持的表情", Object.keys(headMesh.morphTargetDictionary));
      
      if (onHeadMeshReady) {
        onHeadMeshReady(headMesh);
      } 
    } else {
        // 控制台黄色警告
        console.warn('未找到任何包含表情变形的网格，请确认模型是否包含表情预设(BlendShapes)');
    }
  }, [gltf, onHeadMeshReady]);

  return <primitive object={gltf.scene} ref={modelRef} />;
}
export default AvatarModel;