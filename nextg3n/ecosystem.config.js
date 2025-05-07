// /home/ubuntu/nextg3n/nextg3n/ecosystem.config.js
module.exports = {
    apps: [
      {
        name: "nextg3n-scheduler",
        script: "scheduler.py",
        interpreter: "/home/ubuntu/nextg3n/venv/bin/python",
        cwd: "/home/ubuntu/nextg3n/nextg3n",
        autorestart: true,
        watch: false,
        max_memory_restart: "4G",
        log_date_format: "YYYY-MM-DD HH:mm Z"
      },
      {
        name: "nextg3n-dashboard",
        script: "visualization/trade_dashboard.py",
        interpreter: "/home/ubuntu/nextg3n/venv/bin/python",
        cwd: "/home/ubuntu/nextg3n/nextg3n",
        autorestart: true,
        watch: false,
        max_memory_restart: "2G",
        log_date_format: "YYYY-MM-DD HH:mm Z"
      },
      {
        name: "nextg3n-metrics-api",
        script: "visualization/metrics_api.py",
        interpreter: "/home/ubuntu/nextg3n/venv/bin/python",
        cwd: "/home/ubuntu/nextg3n/nextg3n",
        autorestart: true,
        watch: false,
        max_memory_restart: "2G",
        log_date_format: "YYYY-MM-DD HH:mm Z"
      }
    ]
  };