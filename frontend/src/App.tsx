// src/App.tsx
// VRM 头部面部实时驱动系统
// - 摄像机对准头部展示
// - 支持两种模式：
//   ① 本地模拟（LocalArkitsSimulator）：无需后端，循环播放预设表情
//   ② 在线驱动（ArkitsDriver）：WebSocket 接收后端 ARkit 数据
// - VrmModel 加载模型并根据 blendshapes 实时驱动表情 + 头部姿态
import { Suspense, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import VrmModel from './components/VrmModel';
import ArkitsDriver from './components/ArkitsDriver';
import LocalArkitsSimulator from './components/LocalArkitsSimulator';
import type { ArkitsFrame } from './components/VrmModel';

// ── 后端 WebSocket 地址（在线模式时使用）──
const SERVER_URL = 'http://localhost:5000';

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial color="#888" wireframe />
    </mesh>
  );
}

export default function App() {
  // 共享引用：数据源写入最新帧 → VrmModel 每帧消费
  const frameRef = useRef<ArkitsFrame | null>(null);

  // 模式：true = 在线（WebSocket），false = 本地模拟
  const [onlineMode, setOnlineMode] = useState(false);

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, 1.55, 1.0], fov: 30 }}
        style={{ background: '#1a1a1a' }}
      >
        <ambientLight intensity={0.8} />
        <directionalLight position={[1, 2, 3]} intensity={1.0} />

        <Suspense fallback={<LoadingFallback />}>
          <VrmModel
            url="/models/avatar.vrm"
            frameRef={frameRef}
          />
        </Suspense>

        <OrbitControls
          target={[0, 1.55, 0]}
          enablePan={false}
          minDistance={0.5}
          maxDistance={2.0}
        />
      </Canvas>

      {/* ── 模式切换按钮 ── */}
      <div
        style={{
          position: 'absolute',
          bottom: 16,
          left: '50%',
          transform: 'translateX(-50%)',
          background: '#000000aa',
          borderRadius: 8,
          padding: '4px',
          display: 'flex',
          gap: 4,
          fontFamily: 'sans-serif',
          fontSize: '0.85em',
        }}
      >
        <button
          onClick={() => setOnlineMode(false)}
          style={{
            padding: '8px 20px',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
            background: onlineMode ? 'transparent' : '#4caf50',
            color: onlineMode ? '#aaa' : '#fff',
            fontWeight: onlineMode ? 400 : 600,
            transition: 'all 0.2s',
          }}
        >
          🧪 本地模拟
        </button>
        <button
          onClick={() => setOnlineMode(true)}
          style={{
            padding: '8px 20px',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
            background: onlineMode ? '#2196f3' : 'transparent',
            color: onlineMode ? '#fff' : '#aaa',
            fontWeight: onlineMode ? 600 : 400,
            transition: 'all 0.2s',
          }}
        >
          🌐 在线驱动
        </button>
      </div>

      {/* ── 数据源 ── */}
      {onlineMode ? (
        <ArkitsDriver serverUrl={SERVER_URL} frameRef={frameRef} />
      ) : (
        <LocalArkitsSimulator frameRef={frameRef} enabled={true} />
      )}
    </div>
  );
}
