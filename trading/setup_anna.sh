#!/bin/bash
# anna 서버 ai-lab 트레이딩 봇 설치 스크립트
# Jupyter 터미널에서 실행: bash ~/ai-lab/trading/setup_anna.sh

# 1. 의존성 설치
pip install streamlit python-binance python-dotenv pandas_ta stable-baselines3 gymnasium requests

# 2. .env 생성
cat > ~/ai-lab/trading/.env << 'EOF'
# 바이낸스 API
BINANCE_API_KEY=IZABohzjh1kDy66OFjCbQtzguaimEGXhTFT37wWY358vGxhznGeFO8Obg29VY7xT
BINANCE_API_SECRET=vnxPbEVGXpjkxUB5ZG9D4RBDMGvDMr7bqErVjtvShKV9OYxfIBSZDGtJM1z8lQ7C

# 텔레그램
TELEGRAM_TOKEN=8690641013:AAEKBcZbMA4Kl-HLXO_5Zh1cLfk3k31l-W4
TELEGRAM_CHAT_ID=1765732154
EOF

# 3. 실행 (포트는 환경에 맞게 수정)
echo "설치 완료. 아래 명령으로 실행:"
echo "cd ~/ai-lab/trading && streamlit run app.py --server.port 8503"
