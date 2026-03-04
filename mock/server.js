const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
app.use(express.json());

// Загружаем демо-данные
const dataPath = path.join(__dirname, 'data', 'btcusdt_demo.json');
const rawData = JSON.parse(fs.readFileSync(dataPath, 'utf8'));

// Извлекаем символ из первой записи (если есть), иначе используем запасной
const symbol = rawData[0]?.symbol || "UNKNOWN";

// Преобразуем в удобный формат: массив объектов с полями
const historicalData = rawData.map(item => ({
    timestamp: item.timestamp,
    open: item.open,
    high: item.high,
    low: item.low,
    close: item.close,
    volume: item.volume,
    rd_value: item.rd_value
}));

// Переменная для хранения текущего индекса
let currentIndex = 60; // начинаем с 60, чтобы сразу отдавать READY

// Эндпоинт для получения окна фичей
app.get('/api/ml/ds/feature-windows', (req, res) => {
    if (currentIndex < 60) {
        return res.json({
            timeframe: "1m",
            lookbackSteps: 60,
            featureColumns: ["rd_value", "open", "high", "low", "close", "volume"],
            generatedAt: new Date().toISOString(),
            items: [
                {
                    symbol: symbol,  // используем переменную symbol
                    state: "WARMUP",
                    reason: "not_enough_points",
                    pointsCollected: currentIndex,
                    expectedPoints: 60,
                }
            ]
        });
    }

    const start = currentIndex - 60;
    const windowData = historicalData.slice(start, currentIndex);
    
    const features = windowData.map(row => [
        row.rd_value,
        row.open,
        row.high,
        row.low,
        row.close,
        row.volume
    ]);

    const response = {
        timeframe: "1m",
        lookbackSteps: 60,
        featureColumns: ["rd_value", "open", "high", "low", "close", "volume"],
        generatedAt: new Date().toISOString(),
        items: [
            {
                symbol: symbol,  // используем переменную symbol
                state: "READY",
                reason: null,
                pointsCollected: 60,
                expectedPoints: 60,
                windowStartTimestamp: windowData[0].timestamp,
                windowEndTimestamp: windowData[59].timestamp,
                features: features
            }
        ]
    };

    if (currentIndex < historicalData.length) {
        currentIndex++;
    }

    res.json(response);
});

// Эндпоинт приёма сигналов
app.post('/api/signals/ingest', (req, res) => {
    const { symbol, timestamp, signal, price, rating, source } = req.body;
    if (!symbol || !timestamp || !signal || !price || !rating || !source) {
        return res.status(400).json({ error: "Missing required fields" });
    }
    console.log(`\n[PLATFORM] 🚀 ПОЛУЧЕН СИГНАЛ ОТ ML!`);
    console.log(`Торговая пара: ${symbol} | Сигнал: ${signal} | Цена: ${price} | Уверенность: ${(rating*100).toFixed(1)}% | Режим: ${source}`);
    res.status(200).json({ success: true, message: "Signal accepted" });
});

app.listen(3000, () => console.log('✅ Мок-платформа с реальными данными запущена на http://localhost:3000'));