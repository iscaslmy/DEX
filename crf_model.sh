#!/bin/bash
### BEGIN INIT INFO
# Provides:          
# Required-Start:    $local_fs $network
# Required-Stop:     $local_fs
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: CRF_model service
# Description:       CRF_model service
### END INIT INFO

filepath="./crf_routing111.py"

start(){
    nohup python3 $filepath>/dev/null 2>&1 &
    echo 'CRF_model service OK'
}


stop(){
    serverpid=`ps -aux|grep "$filepath"|grep -v grep|awk '{print $2}'`
    kill -9 $serverpid
    echo 'CRF_model stop OK'
}


restart(){
    stop
    echo 'CRF_model stop OK'
    start
    echo 'CRF_model service OK'
}


case $1 in
    start)
    start
    ;;
    stop)
    stop
    ;;
    restart)
    restart
    ;;
    *)
    start
esac