import { useEffect, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { io, Socket } from 'socket.io-client';
import * as THREE from 'three';

interface FaceDriverProps {
  serverUrl: string; //提供表情的Socket地址
  headMesh: THREE.Mesh | null;
}

const FaceDriver: React.FC<FaceDriverProps> = ({ serverUrl, headMesh }) => {
  // WebSocket接受表情数据的频率极高，故使用useRef避免重新渲染
  // 保存最新的表情值的盒子(非数组)
  const latestDataRef = useRef<Record<string, number>>({});

  // 与WebSocket对接
  useEffect(() => {
    const socket: Socket = io(serverUrl, {
      transports: ['websocket'],
    });

    // socket.on监听服务器发来的消息
    // 监听连接事件
    socket.on('connect', () => {
      console.log('[FaceDriver] 面部数据通道已连接');
    });

    // 监听表情数据
    socket.on('face_data', (data: Record<string, number>) => {
      // 后端发来表情数据后，将其存储到ref中
      latestDataRef.current = data;
    });

    // 监听断开连接
    socket.on('disconnect', () => {
      console.warn('[FaceDriver] 面部数据通道断开');
    });

    return () => {
      socket.disconnect();
    };
  }, [serverUrl]);

  // 将表情权重写入头部网络
  useFrame(() => {
    if (!headMesh) return;

    const data = latestDataRef.current;
    const dict = headMesh.morphTargetDictionary;
    const influences = headMesh.morphTargetInfluences;

    if (!dict || !influences) return;

    // 将后端的表情指令翻译，使头像做出对应表情
    // 将对象转化成键值对数组，以便遍历
    for (const [name, weight] of Object.entries(data)) {
      const index = dict[name];
      if (index !== undefined && index < influences.length) {
        influences[index] = weight;
      }
    }
  });
  
  // 负责后台数据同步和表情驱动
  return null;
}

export default FaceDriver;