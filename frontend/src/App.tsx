import { useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import AvatarModel from './components/AvatarModel';
import FaceDriver from './components/FaceDriver';
import * as THREE from 'three';

export default function App() {
  // 头部网络
  const [headMesh, setHeadMesh] = useState<THREE.Mesh | null>(null);

  // 缓存回调，AvatarModel耗能大，需要避免其因 props 变化而重新加载
  const handleHeadMeshReady = useCallback((mesh: THREE.Mesh) => {
    setHeadMesh(mesh);
  }, []);

  return (
    <Canvas
      camera={{ position: [0, 1.5, 2.5], fov: 45 }}
      style={{ width: '100vw', height: '100vh', background: '#1a1a1a' }}
    >
      {/* 环境光 */}
      <ambientLight intensity={0.7} />

      {/* 平行光 */}
      <directionalLight position={[1, 2, 3]} intensity={1.0} /> 

      {/* 加载并3D角色 */}
      <AvatarModel url='/models/avatar.glb' onHeadMeshReady={handleHeadMeshReady} />

      {/* 有头部网格才能连接面部数据，serverUrl为python后端地址 */}
      {headMesh && (
        <FaceDriver serverUrl='http://localhost:5000' headMesh={headMesh} />
      )}

      {/* 鼠标拖拽旋转视角 */}
      <OrbitControls target={[0, 1.5, 0]} />
    </Canvas>
  );
}
