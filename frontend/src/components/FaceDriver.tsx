import { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { io, Socket } from 'socket.io-client';
import * as THREE from 'three';

interface FaceDriverProps {
  serverUrl: string;           // 后端 Socket.IO 服务的地址
  headMesh: THREE.Mesh | null;  // 从 AvatarModel 传来的头部网格
}

const FaceDriver: React.FC<FaceDriverProps> = ({ serverUrl, headMesh }) => {
  // 使用 useRef 存储最新表情数据，避免触发 React 重新渲染
  const latestDataRef = useRef<Record<string, number>>({});

  // 建立 Socket.IO 连接
  useEffect(() => {
    const socket: Socket = io(serverUrl, {
      transports: ['websocket'], // 强制使用 WebSocket（可降级为 polling）
    });

    socket.on('connect', () => {
      console.log('[FaceDriver] Socket 已连接');
    });

    // 监听后端广播的 'face_data' 事件
    socket.on('face_data', (data: Record<string, number>) => {
      // 仅更新 ref，完全不会导致组件重新渲染
      latestDataRef.current = data;
    });

    socket.on('disconnect', () => {
      console.warn('[FaceDriver] Socket 断开');
    });

    // 组件卸载时断开连接
    return () => {
      socket.disconnect();
    };
  }, [serverUrl]);

  // 每一帧（约 60fps）将 ref 中的数据应用到模型的 BlendShapes
  useFrame(() => {
    if (!headMesh) return; // 模型还未准备好

    const data = latestDataRef.current;
    const dict = headMesh.morphTargetDictionary;
    const influences = headMesh.morphTargetInfluences;

    if (!dict || !influences) return;

    // 遍历后端发来的每一个表情键值对
    for (const [name, weight] of Object.entries(data)) {
      const index = dict[name]; // 从字典中找到该表情对应的索引
      if (index !== undefined && index < influences.length) {
        // 直接写入权重值，动画会线性混合
        influences[index] = weight;
      }
    }
  });

  // 该组件不渲染任何图形，只负责逻辑
  return null;
};

export default FaceDriver;