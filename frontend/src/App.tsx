// src/App.tsx
import { useState, Suspense, useRef } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { VRM } from '@pixiv/three-vrm';
import VrmModel from './components/VrmModel';
import OfflineFaceControl, { type FaceControlState } from './components/OfflineFaceControl';
import AnimationController from './components/AnimationController';

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#888" wireframe />
    </mesh>
  );
}

export default function App() {
  // VRM角色
  const [vrm, setVrm] = useState<VRM | null>(null);
  // 表情列表
  const [expressionNames, setExpressionNames] = useState<string[]>([]);

  // ref 桥接：OfflineFaceControl 写入 state → VrmModel 读取
  const faceControlRef = useRef<FaceControlState>({
    weights: {},
  });

  const handleVrmReady = (vrm: VRM) => {
    setVrm(vrm);
    const names: string[] = Object.keys(vrm.expressionManager!.expressionMap!);
    setExpressionNames(names);
  };

  const handleVrmDispose = () => {
    setVrm(null);
    setExpressionNames([]);
  };

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      {/* 将相机放在角色正前方较高处 */}
      <Canvas
        camera={{ position: [0, 1.5, 2.5], fov: 45 }}
        style={{ background: '#1a1a1a' }}
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[1, 2, 3]} intensity={1.0} />

        <Suspense fallback={<LoadingFallback />}>
          <VrmModel
            url="/models/avatar.vrm"
            onVrmReady={handleVrmReady}
            onVrmDispose={handleVrmDispose}
            stateRef={faceControlRef}
          />
        </Suspense>

        <OrbitControls target={[0, 1.5, 0]} />
      </Canvas>

      {/* UI 面板放在 Canvas 外部 */}
      <OfflineFaceControl
        expressionNames={expressionNames}
        stateRef={faceControlRef}
      />
      <AnimationController vrm={vrm} />
    </div>
  );
}
