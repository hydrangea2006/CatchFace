import React, { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { io, Socket } from 'socket.io-client';
import * as THREE from 'three';

interface FaceDriverProps {
  serverUrl: string; // 提供表情数据的 Socket 地址
  headMesh: THREE.Mesh | null;
}

const FaceDriver: React.FC<FaceDriverProps> = ({ serverUrl, headMesh }) => {
  // WebSocket 接受表情数据的频率极高，故使用 useRef 避免重新渲染
  const latestDataRef = useRef<Record<string, number>>({});

  // 与 WebSocket 对接
  useEffect(() => {
    const socket: Socket = io(serverUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });

    socket.on('connect', () => {
      console.log('[FaceDriver] 面部数据通道已连接');
    });

    socket.on('face_data', (data: Record<string, number>) => {
      latestDataRef.current = data;
    });

    socket.on('disconnect', (reason) => {
      console.warn('[FaceDriver] 面部数据通道断开:', reason);
    });

    socket.on('connect_error', (error) => {
      console.error('[FaceDriver] 连接错误:', error.message);
    });

    return () => {
      socket.removeAllListeners();
      socket.disconnect();
    };
  }, [serverUrl]);

  // 将表情权重写入头部网格（每帧执行，与 R3F 渲染循环同步）
  useFrame(() => {
    if (!headMesh) return;

    const data = latestDataRef.current;
    const dict = headMesh.morphTargetDictionary;
    const influences = headMesh.morphTargetInfluences;

    if (!dict || !influences) return;

    for (const [name, weight] of Object.entries(data)) {
      const index = dict[name];
      if (index !== undefined && index < influences.length) {
        influences[index] = weight;
      }
    }
  });

  // 负责后台数据同步和表情驱动，不渲染 UI
  return null;
};

export default FaceDriver;