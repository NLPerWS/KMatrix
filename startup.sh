#!/bin/bash

flask_pids=$(pgrep -f "flask_server.py")
if [ -n "$flask_pids" ]; then
    echo "Stopping Flask server..."
    kill $flask_pids
fi

npm_pids=$(pgrep -f "npm run dev")
if [ -n "$npm_pids" ]; then
    echo "Stopping npm dev server..."
    kill $npm_pids
fi

cd /app/KMatrix
nohup python flask_server.py >log_server.log 2>&1 &

cd easy-flow
mv  /app/node_modules/ ./
nohup npm run dev >log_vue.log 2>&1 &

chmod 777 -R /app/KMatrix

/bin/bash