import network
import time
import json
import urequests
import random
from machine import Pin, ADC, I2C, SPI
from umqtt.simple import MQTTClient
from ssd1306 import SSD1306_I2C
from mcp2515 import MCP2515

# =========================================================
# CONFIG
# =========================================================

SSID = "Project"
PASSWORD = "12345678"

MQTT_HOST = "broker.hivemq.com"
MQTT_PORT = 1883
CLIENT_ID = "pico2w_001"

TOPIC_COOLANT = b"pico/engine/coolant_c"
TOPIC_OIL = b"pico/engine/oil_psi"
TOPIC_MAP = b"pico/engine/map_kpa"
TOPIC_RPM = b"pico/engine/rpm"
TOPIC_STATUS = b"pico/engine/status"

PUBLISH_INTERVAL = 2

API_BASE_URL = "https://lanka-lsub.onrender.com"
API_PREDICT = API_BASE_URL + "/predict"

# CAN IDs
CAN_ID_STATUS = 0x100
CAN_ID_COOLANT = 0x121
CAN_ID_OIL = 0x122
CAN_ID_MAP = 0x123
CAN_ID_RPM = 0x124

# =========================================================
# HARDWARE
# =========================================================

# Pico onboard LED
led = Pin("LED", Pin.OUT)

# Status LEDs
# GP14 = WARN (yellow)
# GP13 = CRITICAL (red)
led_warn = Pin(14, Pin.OUT)
led_critical = Pin(13, Pin.OUT)

# OLED I2C
i2c = I2C(
    0,
    scl=Pin(5),
    sda=Pin(4),
    freq=400000
)

oled = SSD1306_I2C(128, 64, i2c)

# MCP2515 CAN
spi = SPI(
    0,
    baudrate=10000000,
    polarity=0,
    phase=0,
    sck=Pin(18),
    mosi=Pin(19),
    miso=Pin(16)
)

can_cs = Pin(17, Pin.OUT)
can_int = Pin(20, Pin.IN)

# =========================================================
# OLED
# =========================================================

def oled_show(
    line1="",
    line2="",
    line3="",
    line4="",
    line5="",
    line6=""
):
    oled.fill(0)

    oled.text(line1, 0, 0)
    oled.text(line2, 0, 10)
    oled.text(line3, 0, 20)
    oled.text(line4, 0, 30)
    oled.text(line5, 0, 40)
    oled.text(line6, 0, 54)

    oled.show()

# =========================================================
# STATUS LED CONTROL
# =========================================================

def set_status_leds(status):

    s = str(status).lower().strip()

    if s == "critical":
        led_warn.off()
        led_critical.on()

    elif s == "warn":
        led_warn.on()
        led_critical.off()

    else:
        led_warn.off()
        led_critical.off()


def clear_status_leds():
    led_warn.off()
    led_critical.off()

# =========================================================
# RANDOM ENGINE DATA GENERATOR
# =========================================================

# Thresholds
#
# NORMAL
# Coolant : 90 - 105 C
# Oil     : 20 - 60 PSI
# MAP     : 30 - 50 kPa
# RPM     : 600 - 1000
#
# WARN
# Coolant : 108 - 114 C
# Oil     : 10 - 14 PSI
# MAP     : 20 - 26 kPa
# RPM     : 6500 - 7100
#
# CRITICAL
# Coolant : 116 - 130 C
# Oil     : 1 - 9 PSI
# MAP     : 102 - 120 kPa
# RPM     : 7200 - 8000

_ZONE_TABLE = [
    ("normal", 5),
    ("warn", 3),
    ("critical", 2),
]


def _pick_zone():

    total = sum(weight for _, weight in _ZONE_TABLE)

    r = random.randint(0, total - 1)

    acc = 0

    for label, weight in _ZONE_TABLE:

        acc += weight

        if r < acc:
            return label

    return "normal"


def _rnd(low, high, decimals=1):

    value = low + random.random() * (high - low)

    return round(value, decimals)


def generate_engine_data():

    def coolant(zone):

        if zone == "critical":
            return _rnd(116.0, 130.0)

        if zone == "warn":
            return _rnd(108.0, 114.9)

        return _rnd(90.0, 105.0)

    def oil(zone):

        if zone == "critical":
            return _rnd(1.0, 9.9)

        if zone == "warn":
            return _rnd(10.0, 14.9)

        return _rnd(20.0, 60.0)

    def map_kpa(zone):

        if zone == "critical":
            return _rnd(102.0, 120.0)

        if zone == "warn":
            return _rnd(20.0, 26.9)

        return _rnd(30.0, 50.0)

    def rpm(zone):

        if zone == "critical":
            return int(_rnd(7200, 8000, 0))

        if zone == "warn":
            return int(_rnd(6500, 7100, 0))

        return int(_rnd(600, 1000, 0))

    coolant_c = coolant(_pick_zone())
    oil_psi = oil(_pick_zone())
    map_kpa = map_kpa(_pick_zone())
    rpm_value = rpm(_pick_zone())

    print(
        "[GEN] "
        "CLT={}C "
        "OIL={}psi "
        "MAP={}kPa "
        "RPM={}".format(
            coolant_c,
            oil_psi,
            map_kpa,
            rpm_value
        )
    )

    return coolant_c, oil_psi, map_kpa, rpm_value

# =========================================================
# STATUS NORMALIZER
# =========================================================

def normalize_status(raw):

    s = str(raw).upper().strip()

    if s == "GOOD":
        return "ok"

    if s == "WARN":
        return "warn"

    if s == "CRIT":
        return "critical"

    return "ok"

# =========================================================
# PREDICT API
# =========================================================

def predict_status():

    coolant_c, oil_psi, map_kpa, rpm = generate_engine_data()

    payload = json.dumps({
        "coolant_c": coolant_c,
        "oil_psi": oil_psi,
        "map_kpa": map_kpa,
        "rpm": rpm
    })

    try:

        res = urequests.post(
            API_PREDICT,
            data=payload,
            headers={
                "Content-Type": "application/json"
            },
            timeout=8
        )

        if res.status_code == 200:

            body = res.json()

            res.close()

            # YOUR API FORMAT
            #
            # {
            #   "prediction": {
            #       "status": "GOOD",
            #       "confidence": {
            #           "CRIT": 0.0,
            #           "GOOD": 1.0,
            #           "WARN": 0.0
            #       }
            #   }
            # }

            prediction = body.get("prediction", {})

            raw_status = prediction.get("status", "GOOD")

            confidence = prediction.get(
                "confidence",
                {}
            )

            status = normalize_status(raw_status)

            print("[PREDICT] STATUS:", raw_status)
            print("[PREDICT] NORMALIZED:", status)
            print("[PREDICT] CONFIDENCE:", confidence)

            return (
                coolant_c,
                oil_psi,
                map_kpa,
                rpm,
                status,
                confidence
            )

        else:

            print(
                "[PREDICT] HTTP ERROR:",
                res.status_code
            )

            res.close()

            return (
                coolant_c,
                oil_psi,
                map_kpa,
                rpm,
                "ok",
                {}
            )

    except Exception as e:

        print("[PREDICT] ERROR:", e)

        return (
            coolant_c,
            oil_psi,
            map_kpa,
            rpm,
            "ok",
            {}
        )

# =========================================================
# WIFI
# =========================================================

def connect_wifi():

    wlan = network.WLAN(network.STA_IF)

    wlan.active(True)

    oled_show(
        "WiFi",
        "Connecting..."
    )

    print("Connecting WiFi...")

    if not wlan.isconnected():

        wlan.connect(
            SSID,
            PASSWORD
        )

        timeout = 15

        while (
            not wlan.isconnected()
            and timeout > 0
        ):

            print(".", end="")

            time.sleep(1)

            timeout -= 1

    if wlan.isconnected():

        ip = wlan.ifconfig()[0]

        print("\nWiFi OK:", ip)

        oled_show(
            "WiFi OK",
            ip
        )

        time.sleep(1)

        return wlan

    oled_show(
        "WiFi Failed"
    )

    raise OSError("WiFi failed")

# =========================================================
# MQTT
# =========================================================

def connect_mqtt():

    oled_show(
        "MQTT",
        "Connecting..."
    )

    print("Connecting MQTT...")

    client = MQTTClient(
        CLIENT_ID,
        MQTT_HOST,
        port=MQTT_PORT,
        keepalive=60
    )

    client.connect()

    client.publish(
        TOPIC_STATUS,
        b"online"
    )

    print("MQTT Connected")

    oled_show(
        "MQTT OK"
    )

    time.sleep(1)

    return client

# =========================================================
# CAN
# =========================================================

def connect_can():

    oled_show(
        "CAN",
        "Starting..."
    )

    print("Starting MCP2515...")

    can = MCP2515(
        spi,
        can_cs
    )

    can.set_bitrate(500000)

    can.set_normal_mode()

    print("CAN OK")

    oled_show(
        "CAN OK",
        "500kbps"
    )

    time.sleep(1)

    return can


def pack_value(value):

    value = int(value * 100)

    value = max(
        0,
        min(value, 65535)
    )

    high = (value >> 8) & 0xFF
    low = value & 0xFF

    return high, low


def send_can_single_value(
    can,
    can_id,
    value
):

    high, low = pack_value(value)

    data = bytearray([
        high,
        low,
        0,
        0,
        0,
        0,
        0,
        0
    ])

    can.send(can_id, data)


def send_can_engine_data(
    can,
    coolant_c,
    oil_psi,
    map_kpa,
    rpm
):

    send_can_single_value(
        can,
        CAN_ID_COOLANT,
        coolant_c
    )

    send_can_single_value(
        can,
        CAN_ID_OIL,
        oil_psi
    )

    send_can_single_value(
        can,
        CAN_ID_MAP,
        map_kpa
    )

    send_can_single_value(
        can,
        CAN_ID_RPM,
        rpm
    )

# =========================================================
# MAIN
# =========================================================

oled_show(
    "Pico 2 W",
    "Starting..."
)

clear_status_leds()

connect_wifi()

led.on()

client = connect_mqtt()

can = connect_can()

print("\nSystem Running...\n")

while True:

    try:

        (
            coolant_c,
            oil_psi,
            map_kpa,
            rpm,
            status,
            confidence
        ) = predict_status()

        # =========================================
        # LEDs
        # =========================================

        set_status_leds(status)

        # =========================================
        # MQTT
        # =========================================

        client.publish(
            TOPIC_COOLANT,
            str(coolant_c).encode()
        )

        client.publish(
            TOPIC_OIL,
            str(oil_psi).encode()
        )

        client.publish(
            TOPIC_MAP,
            str(map_kpa).encode()
        )

        client.publish(
            TOPIC_RPM,
            str(rpm).encode()
        )

        client.publish(
            TOPIC_STATUS,
            str(status).encode()
        )

        # =========================================
        # CAN
        # =========================================

        send_can_engine_data(
            can,
            coolant_c,
            oil_psi,
            map_kpa,
            rpm
        )

        # =========================================
        # OLED
        # =========================================

        oled_show(
            "Engine Data",
            "CLT:{}C".format(coolant_c),
            "OIL:{}psi".format(oil_psi),
            "MAP:{}kPa".format(map_kpa),
            "RPM:{}".format(rpm),
            "ST:{}".format(
                status.upper()
            )
        )

        # =========================================
        # SERIAL PRINT
        # =========================================

        print(
            "CLT:{}C | "
            "OIL:{}psi | "
            "MAP:{}kPa | "
            "RPM:{} | "
            "STATUS:{} | "
            "CONF:{}".format(
                coolant_c,
                oil_psi,
                map_kpa,
                rpm,
                status.upper(),
                confidence
            )
        )

        time.sleep(PUBLISH_INTERVAL)

    except OSError as e:

        print(
            "Connection Lost:",
            e
        )

        led.off()

        clear_status_leds()

        oled_show(
            "Reconnect...",
            str(e)
        )

        time.sleep(3)

        try:

            connect_wifi()

            client = connect_mqtt()

            can = connect_can()

            led.on()

        except Exception as ex:

            print(
                "Reconnect Failed:",
                ex
            )

            oled_show(
                "Reconnect Fail",
                str(ex)
            )

            time.sleep(5)
