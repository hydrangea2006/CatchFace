// server.js
const http = require('http');
const socketIo = require('socket.io');

const server = http.createServer();
const io = socketIo(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});

io.on('connection', (socket) => {
    console.log(`[客户端连接] ${socket.id}`);

    // 接收面部数据
    socket.on('blendshapes', (data) => {
        console.log('\n========== 接收到面部数据 ==========');
        console.log(`时间戳: ${data.timestamp}`);
        console.log(`头部旋转: X=${data.head.rotation.x.toFixed(3)}, Y=${data.head.rotation.y.toFixed(3)}, Z=${data.head.rotation.z.toFixed(3)}`);
        console.log(`头部位置: X=${data.head.position[0].toFixed(3)}, Y=${data.head.position[1].toFixed(3)}, Z=${data.head.position[2].toFixed(3)}`);
        console.log(`Blendshapes (前10个):`);

        // 打印前10个 blendshape
        const blendshapes = data.blendshapes;
        const entries = Object.entries(blendshapes).slice(0, 10);
        entries.forEach(([key, value]) => {
            console.log(`  ${key}: ${value.toFixed(4)}`);
        });
        console.log('=====================================\n');
    });

    socket.on('disconnect', () => {
        console.log(`[客户端断开] ${socket.id}`);
    });
});

const PORT = 5000;
server.listen(PORT, () => {
    console.log(`服务器运行在 http://localhost:${PORT}`);
    console.log('等待接收面部捕捉数据...\n');
});