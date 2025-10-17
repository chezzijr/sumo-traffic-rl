#!/usr/bin/bash
rm -f vn.*.trips.*

python randomTrips.py -n vn.net.xml -e 3600 -p 0.5 --trip-attributes "type='motorcycle'" --prefix "motorcycle_" -o vn.motorcycle.trips.xml
python randomTrips.py -n vn.net.xml -e 3600 -p 4.0 --trip-attributes "type='car'" --prefix "car_" -o vn.car.trips.xml
python randomTrips.py -n vn.net.xml -e 3600 -p 20.0 --trip-attributes "type='delivery'" --prefix "delivery_" -o vn.delivery.trips.xml
python randomTrips.py -n vn.net.xml -e 3600 -p 30.0 --trip-attributes "type='bus'" --prefix "bus_" -o vn.bus.trips.xml
python randomTrips.py -n vn.net.xml -e 3600 -p 40.0 --trip-attributes 'type="truck"' --vclass truck --edge-permission truck --prefix "truck_" --fringe-factor 5 -o vn.truck.trips.xml
