import { useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import AvatarModel from './components/AvatarModel';
import FaceDriver from './components/FaceDriver';
import * as THREE from 'three';

function App() {
  // 存储从 AvatarModel 获得的头部 Mesh，初始为 null
  const [headMesh, setHeadMesh] = useState<THREE.Mesh | null>(null);
  
  // 组件每次刷新时都使用这同一函数,而不是创建一个新的函数
  // 用 useCallback 包裹回调，避免不必要的重渲染
  const handleHeadMeshReady = useCallback((mesh: THREE.Mesh) => {
    setHeadMesh(mesh);
  }, []);

  return (
    // Canvas 组件会创建一个 Three.js 渲染器，并自动处理 resize
    <Canvas
      camera={{ position: [0, 1.5, 2.5], fov: 45 }}
      style={{ width: '100vw', height: '100vh', background: '#1a1a1a' }}
    >
      {/* 环境光：柔和照亮模型 */}
      <ambientLight intensity={0.7} />
      {/* 平行光：模拟主光源，产生立体感 */}
      <directionalLight position={[1, 2, 3]} intensity={1.0} />

      {/* 加载 Ready Player Me 模型，并接收头部 Mesh */}
      <AvatarModel url="/models/avatar.glb" onHeadMeshReady={handleHeadMeshReady} />

      {/* 只有当头部 Mesh 存在时才挂载驱动组件，避免无效操作 */}
      {headMesh && (
        <FaceDriver serverUrl="http://localhost:5000" headMesh={headMesh} />
      )}

      {/* 轨道控制器：允许用户用鼠标旋转、缩放视角（调试用） */}
      <OrbitControls target={[0, 1.5, 0]} />
    </Canvas>
  );
}

export default App;