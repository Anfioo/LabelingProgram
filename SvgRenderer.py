import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon, QPixmap, QPainter
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtCore import QByteArray, Qt


def set_svg_icon_from_string(window, svg_string):
    """将 SVG 字符串设置为窗口图标"""
    svg_data = QByteArray(svg_string.encode('utf-8'))
    renderer = QSvgRenderer(svg_data)

    # 创建适当大小的 pixmap
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)

    # 渲染 SVG 到 pixmap
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    window.setWindowIcon(QIcon(pixmap))


def get_config_svg_icon():
    icon_string = """
        <svg t="1744263424791" class="icon" viewBox="0 0 1062 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4648" width="200" height="200"><path d="M708.719169 393.40363c-142.527361 2.733235-255.748369 115.555104-256.997848 257.015202-3.353636 141.585914 115.715627 251.514024 270.039284 260.399207 154.523226 10.33857 255.297168-94.695748 244.962936-249.483621-8.650906-154.662057-115.268765-272.382056-258.004372-267.930788z" fill="#FFCA42" p-id="4649"></path><path d="M491.175352 109.511617c-153.403901-2.234311-261.045639 101.967021-253.266765 247.626756a294.365075 294.365075 0 0 0 14.633654 80.383141c5.427424 16.403749 22.49062 25.49284 38.26529 20.412494a29.210907 29.210907 0 0 0 16.703103-13.82236c70.01854 41.219787 138.982833 80.55668 208.363618 120.162557a29.948447 29.948447 0 0 0-3.301574 21.453726c3.106343 14.017591 15.601132 23.56656 29.471215 23.588252a30.000508 30.000508 0 0 0 6.581457-0.715847c55.467317-12.147711 105.303303-42.738251 141.008897-87.346383 36.22621-45.198163 56.864304-103.242531 58.881692-162.267392 6.594472-143.611978-103.975732-248.641958-257.340587-249.474944z" fill="#FF7575" p-id="4650"></path><path d="M336.934126 402.679275c-142.523023-4.256037-249.483621 113.884793-258.763604 268.542512-10.963309 154.792211 89.762909 259.401359 244.858814 248.871897 154.896334-9.089091 274.312675-118.917417 270.802853-260.512007-1.401325-141.455759-114.578948-254.373074-256.898063-256.902402z" fill="#78BEFF" p-id="4651"></path></svg>
        """
    return icon_string


def get_main_svg_icon():
    icon_string = """
      <svg t="1744263587058" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="11493" width="200" height="200"><path d="M512 256c-141.4 0-256 114.6-256 256s114.6 256 256 256 256-114.6 256-256-114.6-256-256-256z m0 298.7c-23.6 0-42.7-19.1-42.7-42.7s19.1-42.7 42.7-42.7 42.7 19.1 42.7 42.7-19.1 42.7-42.7 42.7z" fill="#F9DB88" p-id="11494"></path><path d="M512 85.3c-235.6 0-426.7 191-426.7 426.7s191 426.7 426.7 426.7 426.7-191 426.7-426.7S747.6 85.3 512 85.3z m42.7 765.3v-96.8c0-23.6-19.1-42.7-42.7-42.7s-42.7 19.1-42.7 42.7v96.8c-154.1-19.3-276.6-141.8-295.9-295.9h96.8c23.6 0 42.7-19.1 42.7-42.7s-19.1-42.7-42.7-42.7h-96.8c19.3-154.1 141.8-276.6 295.9-295.9v96.8c0 23.6 19.1 42.7 42.7 42.7s42.7-19.1 42.7-42.7v-96.8c154.1 19.3 276.6 141.8 295.9 295.9h-96.8c-23.6 0-42.7 19.1-42.7 42.7s19.1 42.7 42.7 42.7h96.8c-19.4 154.1-141.8 276.5-295.9 295.9z" fill="#009F72" p-id="11495"></path></svg>
       """
    return icon_string


def get_setting_svg_icon():
    icon_string = """
    <svg t="1744286652801" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4461" width="200" height="200"><path d="M438.109867 245.191111a25.258667 25.258667 0 0 0-9.955556 20.081778c0 11.150222 5.12 21.731556 13.994667 28.273778l38.855111 29.354666a22.528 22.528 0 0 1 8.533333 21.560889l-0.227555 1.649778a27.420444 27.420444 0 0 1-0.284445 1.820444c-1.194667 7.452444-8.476444 28.615111-21.788444 53.816889-13.255111 25.144889-28.444444 39.936-33.109334 44.487111l-1.137777 0.910223a25.315556 25.315556 0 0 1-26.567111 3.697777l-46.193778-20.650666a28.842667 28.842667 0 0 0-24.746667 1.308444 21.845333 21.845333 0 0 0-11.377778 17.123556l-6.826666 56.888889a19.626667 19.626667 0 0 1-13.084445 16.497777c-1.877333 0.682667-3.811556 1.080889-5.688889 1.592889h-0.568889c-0.284444 0.170667-0.682667 0.170667-1.080888 0.398223a212.195556 212.195556 0 0 1-51.939556 6.542222c-24.860444 0-43.690667-4.551111-51.939556-6.542222-0.398222-0.227556-0.739556-0.227556-1.137777-0.398223h-0.455111c-1.934222-0.568889-3.811556-0.910222-5.688889-1.649777a19.512889 19.512889 0 0 1-13.084445-16.384L165.953422 448.284444a26.908444 26.908444 0 0 0-13.255111-19.512888 22.869333 22.869333 0 0 0-21.162667-1.877334l-53.304888 23.893334a16.554667 16.554667 0 0 1-16.839112-2.332445l-3.015111-2.389333a31.516444 31.516444 0 0 1-5.916444-5.12 285.923556 285.923556 0 0 1-29.752889-41.756445C5.526756 368.924444 0.634311 346.282667 0.634311 343.722667a35.328 35.328 0 0 1 13.653333-32.426667l37.432889-28.330667a25.201778 25.201778 0 0 0 0-40.334222l-42.837333-32.426667A22.186667 22.186667 0 0 1 0.520533 187.733333c0.113778-0.739556 0.227556-1.649778 0.455111-2.56 1.536-6.200889 8.760889-26.794667 21.617778-51.256889 17.123556-30.264889 31.800889-45.454222 34.247111-47.957333a25.258667 25.258667 0 0 1 26.624-3.697778l48.014223 21.560889c6.769778 3.072 14.791111 1.991111 21.162666-1.991111a26.567111 26.567111 0 0 0 13.255111-19.342222L172.666311 25.031111a19.512889 19.512889 0 0 1 13.084445-16.270222 31.687111 31.687111 0 0 1 5.632-1.649778c0.170667 0 0.341333 0 0.512-0.170667 0.341333 0 0.682667-0.170667 1.137777-0.170666C201.167644 4.721778 219.997867 0 244.858311 0c24.860444 0 43.804444 4.721778 51.939556 6.769778 0.398222 0 0.796444 0.170667 1.137777 0.170666a32.142222 32.142222 0 0 1 6.200889 1.820445 19.512889 19.512889 0 0 1 13.084445 16.213333l6.656 57.457778c0.967111 8.248889 6.257778 15.189333 13.312 19.342222 6.257778 4.039111 14.392889 5.12 21.162666 2.048l48.014223-21.560889a25.315556 25.315556 0 0 1 27.704889 4.778667c4.266667 4.209778 17.692444 17.123556 33.109333 44.373333 17.066667 30.435556 22.016 53.020444 22.016 55.580445a30.151111 30.151111 0 0 1-11.662222 28.273778L438.109867 245.191111zM245.825422 329.671111c31.857778 0 57.742222-26.282667 57.742222-58.766222 0-32.256-25.827556-58.538667-57.742222-58.538667-31.914667 0-57.799111 26.282667-57.799111 58.595556 0 32.426667 25.884444 58.709333 57.799111 58.709333z" fill="#04BABE" p-id="4462"></path><path d="M920.698311 473.144889a48.241778 48.241778 0 0 0-19.911111 38.855111c0 21.447111 10.353778 41.756444 27.875556 54.442667l77.767111 56.832c13.084444 9.557333 19.683556 25.6 17.066666 41.358222l-0.455111 3.527111-0.512 3.299556c-2.389333 14.449778-16.952889 55.068444-43.576889 103.879111a333.027556 333.027556 0 0 1-66.218666 85.674666l-2.218667 2.161778a52.508444 52.508444 0 0 1-53.248 6.997333l-92.330667-39.992888c-15.815111-6.826667-34.133333-5.290667-49.550222 2.332444a42.439111 42.439111 0 0 0-22.584889 33.393778l-13.767111 109.852444a38.4 38.4 0 0 1-26.225778 31.630222c-3.697778 1.137778-7.566222 2.161778-11.377777 3.128889a87.438222 87.438222 0 0 1-3.242667 0.796445c-16.384 3.697778-54.101333 12.686222-103.879111 12.686222-49.720889 0-87.438222-8.988444-103.822222-12.686222a59.733333 59.733333 0 0 1-2.275556-0.568889c-0.341333-0.227556-0.682667-0.227556-1.024-0.227556-3.811556-0.967111-7.623111-1.934222-11.320889-3.128889a38.229333 38.229333 0 0 1-26.168889-31.630222l-13.425778-110.648889c-1.934222-16.213333-12.515556-29.070222-26.624-37.489777a46.933333 46.933333 0 0 0-42.211555-3.868445l-106.666667 46.250667a33.336889 33.336889 0 0 1-33.678222-4.551111 81.351111 81.351111 0 0 0-5.973333-4.437334c-4.152889-3.128889-8.362667-6.257778-11.832889-9.784889-12.174222-12.686222-34.588444-41.528889-59.562667-80.782222-34.190222-58.368-43.975111-102.286222-43.975111-107.178666-3.015111-24.177778 7.395556-48.014222 27.306667-62.464l74.865777-54.613334a48.355556 48.355556 0 0 0 19.968-39.082666 48.241778 48.241778 0 0 0-19.911111-38.855112L62.188089 405.788444a42.837333 42.837333 0 0 1-17.066667-41.528888l0.284445-1.991112c0.284444-1.536 0.512-3.128889 0.910222-4.835555 3.128889-12.117333 17.578667-51.768889 43.349333-98.986667 34.190222-58.595556 63.488-87.836444 68.380445-92.728889a52.508444 52.508444 0 0 1 53.248-6.997333l96.142222 41.528889c13.425778 5.859556 29.582222 3.754667 42.154667-3.697778 14.108444-8.362667 24.689778-21.447111 26.624-37.489778L389.754311 48.241778a38.684444 38.684444 0 0 1 26.112-31.630222A113.208889 113.208889 0 0 1 427.1872 13.653333c0.341333-0.170667 0.682667-0.170667 1.024-0.170666A434.062222 434.062222 0 0 1 534.308978 0a434.403556 434.403556 0 0 1 107.178666 13.653333c3.754667 0.796444 7.623111 1.763556 11.320889 2.958223 14.165333 4.835556 24.348444 17.180444 26.168889 31.630222l13.425778 110.819555c1.934222 16.042667 12.515556 29.127111 26.567111 37.546667 12.629333 7.395556 28.785778 9.500444 42.211556 3.640889l96.142222-41.528889a52.508444 52.508444 0 0 1 53.191111 6.997333c0.512 0.568889 1.251556 1.137778 2.161778 2.161778 8.647111 7.964444 35.441778 32.995556 66.275555 85.674667 34.190222 58.595556 43.975111 102.456889 43.975111 107.349333a57.457778 57.457778 0 0 1-23.324444 54.670222l-78.904889 57.571556zM535.958756 636.359111c63.829333 0 115.484444-50.744889 115.484444-113.436444 0-62.464-51.655111-113.208889-115.484444-113.208889-63.829333 0-115.484444 50.744889-115.484445 113.208889 0 62.691556 51.655111 113.436444 115.484445 113.436444z" fill="#B4EBED" p-id="4463"></path></svg>
     """
    return icon_string


def get_shortcuts_svg_icon():
    icon_string = """
    <svg t="1744288297316" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="9108" width="200" height="200"><path d="M908.544 156.16H141.056c-30.6944 0-57.6512 11.0592-79.5648 32.6144-22.016 21.6576-33.3312 48.384-33.3312 78.848v488.7808c0 30.464 11.3152 57.1648 33.3312 78.8224A110.2848 110.2848 0 0 0 141.056 867.84h767.488c30.72 0 57.6512-11.0592 79.5648-32.6144 22.016-21.6576 33.3312-48.3584 33.3312-78.8224V267.6224c0-30.464-11.3152-57.1904-33.3312-78.848A110.208 110.208 0 0 0 908.5696 156.16zM141.056 202.24h767.488c18.6112 0 33.9456 6.2976 47.2576 19.4048 13.2352 13.0048 19.5584 27.904 19.5584 45.9776v488.7808c0 18.048-6.3232 32.9216-19.5584 45.952-13.312 13.1072-28.672 19.4048-47.232 19.4048H141.056a64.256 64.256 0 0 1-47.2832-19.4048c-13.2352-13.0304-19.5584-27.904-19.5584-45.952V267.6224c0-18.048 6.3232-32.9728 19.5584-45.9776A64.256 64.256 0 0 1 141.056 202.24z" fill="#666666" p-id="9109"></path><path d="M281.6 476.16a23.04 23.04 0 0 1 2.8928 45.9008L281.6 522.24H204.8a23.04 23.04 0 0 1-2.8928-45.9008L204.8 476.16h76.8z" fill="#666666" p-id="9110"></path><path d="M460.8 476.16a23.04 23.04 0 0 1 2.8928 45.9008L460.8 522.24h-76.8a23.04 23.04 0 0 1-2.8928-45.9008L384 476.16h76.8z" fill="#FF6D00" p-id="9111"></path><path d="M640 476.16a23.04 23.04 0 0 1 2.8928 45.9008L640 522.24h-76.8a23.04 23.04 0 0 1-2.8928-45.9008L563.2 476.16h76.8zM844.8 629.76a23.04 23.04 0 0 1 2.8928 45.9008L844.8 675.84H204.8a23.04 23.04 0 0 1-2.8928-45.9008L204.8 629.76h640zM819.2 476.16a23.04 23.04 0 0 1 2.8928 45.9008L819.2 522.24h-76.8a23.04 23.04 0 0 1-2.8928-45.9008L742.4 476.16h76.8zM281.6 348.16a23.04 23.04 0 0 1 2.8928 45.9008L281.6 394.24H204.8a23.04 23.04 0 0 1-2.8928-45.9008L204.8 348.16h76.8zM460.8 348.16a23.04 23.04 0 0 1 2.8928 45.9008L460.8 394.24h-76.8a23.04 23.04 0 0 1-2.8928-45.9008L384 348.16h76.8z" fill="#666666" p-id="9112"></path><path d="M640 348.16a23.04 23.04 0 0 1 2.8928 45.9008L640 394.24h-76.8a23.04 23.04 0 0 1-2.8928-45.9008L563.2 348.16h76.8z" fill="#FF6D00" p-id="9113"></path><path d="M819.2 348.16a23.04 23.04 0 0 1 2.8928 45.9008L819.2 394.24h-76.8a23.04 23.04 0 0 1-2.8928-45.9008L742.4 348.16h76.8z" fill="#666666" p-id="9114"></path></svg>
    """
    return icon_string


def get_check_svg_icon():
    icon_string = """
   <svg t="1744300763828" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="30853" width="200" height="200"><path d="M319.9 310c-11 0-20-9-20-20v-20c0-11 9-20 20-20H704c11 0 20 9 20 20v20c0 11-9 20-20 20M556.1 463H319.9c-11 0-20-9-20-20v-20c0-11 9-20 20-20h236.2c11 0 20 9 20 20v20c0 11.1-8.9 20-20 20zM404.1 619H320c-11 0-20-9-20-20v-20c0-11 9-20 20-20h84.1c11 0 20 9 20 20v20c0 11.1-9 20-20 20z" fill="#666666" p-id="30854"></path><path d="M765.1 957.1H258.9c-71.7 0-130-58.3-130-130V196.9c0-71.7 58.3-130 130-130h506.2c71.7 0 130 58.3 130 130V827c0 71.8-58.3 130.1-130 130.1zM258.9 126.9c-38.6 0-70 31.4-70 70V827c0 38.6 31.4 70 70 70h506.2c38.6 0 70-31.4 70-70V196.9c0-38.6-31.4-70-70-70H258.9z" fill="#666666" p-id="30855"></path><path d="M707.1 769l-58.3-63.1c-7.6-8-7.4-20.6 0.6-28.3l14.4-13.8c8-7.6 20.6-7.4 28.3 0.6l58.3 63.1c7.6 8 7.4 20.6-0.6 28.3l-14.4 13.8c-8 7.7-20.6 7.4-28.3-0.6z" fill="#2AACB2" p-id="30856"></path><path d="M595.1 742c-72.7 0-131.8-59.1-131.8-131.8s59.1-131.8 131.8-131.8 131.8 59.1 131.8 131.8S667.8 742 595.1 742z m0-203.7c-39.6 0-71.8 32.2-71.8 71.8s32.2 71.8 71.8 71.8 71.8-32.2 71.8-71.8-32.2-71.8-71.8-71.8z" fill="#2AACB2" p-id="30857"></path></svg>
   """
    return icon_string


def get_deal_svg_icon():
    icon_string = """
   <svg t="1744300806056" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="35521" width="200" height="200"><path d="M728.615385 315.076923h-472.615385c-19.692308 0-39.384615-15.753846-39.384615-39.384615 0-19.692308 15.753846-39.384615 39.384615-39.384616h472.615385c23.630769 0 39.384615 15.753846 39.384615 39.384616s-19.692308 39.384615-39.384615 39.384615zM555.323077 555.323077H256c-19.692308 0-39.384615-15.753846-39.384615-39.384615s15.753846-39.384615 39.384615-39.384616h299.323077c19.692308 0 39.384615 15.753846 39.384615 39.384616 0 19.692308-19.692308 39.384615-39.384615 39.384615z" fill="#5060B5" p-id="35522"></path><path d="M984.615385 610.461538c-11.815385 0-19.692308 3.938462-27.569231 11.815385l-334.769231 334.769231c-15.753846 15.753846-15.753846 39.384615 0 55.138461 7.876923 7.876923 19.692308 11.815385 27.569231 11.815385s19.692308-3.938462 27.569231-11.815385l334.76923-334.76923c15.753846-15.753846 15.753846-39.384615 0-55.138462-7.876923-7.876923-19.692308-11.815385-27.56923-11.815385z" fill="#FC8739" p-id="35523"></path><path d="M78.769231 827.076923V196.923077c0-66.953846 51.2-118.153846 118.153846-118.153846h630.153846c66.953846 0 118.153846 51.2 118.153846 118.153846v263.876923c0 23.630769 15.753846 39.384615 39.384616 39.384615s39.384615-15.753846 39.384615-39.384615V157.538462c0-86.646154-70.892308-157.538462-157.538462-157.538462H157.538462C70.892308 0 0 70.892308 0 157.538462v708.923076c0 86.646154 70.892308 157.538462 157.538462 157.538462h145.723076c23.630769 0 39.384615-15.753846 39.384616-39.384615s-15.753846-39.384615-39.384616-39.384616H196.923077c-66.953846 0-118.153846-51.2-118.153846-118.153846z" fill="#5060B5" p-id="35524"></path><path d="M468.676923 760.123077c-11.815385 0-19.692308 3.938462-27.569231 11.815385-15.753846 15.753846-15.753846 39.384615 0 55.138461l181.169231 181.169231c7.876923 7.876923 19.692308 11.815385 27.569231 11.815384s19.692308-3.938462 27.569231-11.815384c15.753846-15.753846 15.753846-39.384615 0-55.138462l-181.169231-181.16923c-7.876923-7.876923-19.692308-11.815385-27.569231-11.815385z" fill="#FC8739" p-id="35525"></path></svg>
   """
    return icon_string


def get_show_svg_icon():
    icon_string = """
 <svg t="1744301058463" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="45376" width="200" height="200"><path d="M812.2368 236.9536c13.5168 0 24.576 11.0592 24.576 24.576v369.4592c0 13.5168-11.0592 24.576-24.576 24.576H211.7632c-13.5168 0-24.576-11.0592-24.576-24.576V261.5296c0-13.5168 11.0592-24.576 24.576-24.576h600.4736m0-45.056H211.7632c-38.5024 0-69.632 31.1296-69.632 69.632v369.4592c0 38.5024 31.1296 69.632 69.632 69.632h600.6784c38.5024 0 69.632-31.1296 69.632-69.632V261.5296c-0.2048-38.2976-31.3344-69.632-69.8368-69.632zM164.6592 787.0464h694.8864c12.4928 0 22.528 10.0352 22.528 22.528s-10.0352 22.528-22.528 22.528H164.6592c-12.4928 0-22.528-10.0352-22.528-22.528s10.0352-22.528 22.528-22.528z" fill="#260A50" p-id="45377"></path><path d="M380.928 311.5008h352.256c12.4928 0 22.528 10.0352 22.528 22.528s-10.0352 22.528-22.528 22.528H380.928c-12.4928 0-22.528-10.0352-22.528-22.528 0-12.288 10.24-22.528 22.528-22.528z" fill="#260A50" p-id="45378"></path><path d="M290.816 334.0288m-22.528 0a22.528 22.528 0 1 0 45.056 0 22.528 22.528 0 1 0-45.056 0Z" fill="#260A50" p-id="45379"></path><path d="M380.928 424.1408h352.256c12.4928 0 22.528 10.0352 22.528 22.528s-10.0352 22.528-22.528 22.528H380.928c-12.4928 0-22.528-10.0352-22.528-22.528s10.24-22.528 22.528-22.528z" fill="#F04FC0" p-id="45380"></path><path d="M290.816 446.6688m-22.528 0a22.528 22.528 0 1 0 45.056 0 22.528 22.528 0 1 0-45.056 0Z" fill="#260A50" p-id="45381"></path><path d="M380.928 536.576h352.256c12.4928 0 22.528 10.0352 22.528 22.528s-10.0352 22.528-22.528 22.528H380.928c-12.4928 0-22.528-10.0352-22.528-22.528s10.24-22.528 22.528-22.528z" fill="#260A50" p-id="45382"></path><path d="M290.816 559.104m-22.528 0a22.528 22.528 0 1 0 45.056 0 22.528 22.528 0 1 0-45.056 0Z" fill="#260A50" p-id="45383"></path></svg>
  """
    return icon_string


def get_introduce_svg_icon():
    icon_string = """
<svg t="1744341553354" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="9141" width="200" height="200"><path d="M628.48 394.2912c0-57.1904-48.7424-103.7312-108.6976-103.7312s-108.6976 46.5408-108.6976 103.7312 48.7424 103.7312 108.6976 103.7312 108.6976-46.5408 108.6976-103.7312z" fill="#F66A68" p-id="9142"></path><path d="M515.6352 72.6016c-241.0496 0-436.48 195.4304-436.48 436.48s195.4304 436.48 436.48 436.48 436.48-195.4304 436.48-436.48-195.4304-436.48-436.48-436.48z m209.5104 680.7552c-15.872 0-28.7232-12.8512-28.7232-28.7232 0-93.3376-79.2576-169.216-176.64-169.216s-176.64 75.9296-176.64 169.216c0 15.872-12.8512 28.7232-28.7232 28.7232s-28.7232-12.8512-28.7232-28.7232c0-89.3952 53.76-166.8608 131.584-203.7248C378.5728 491.3664 353.6896 445.5936 353.6896 394.24c0-88.832 74.496-161.1264 166.0928-161.1264 91.5968 0 166.0928 72.2944 166.0928 161.1264 0 51.3536-24.9344 97.1264-63.5904 126.6688 77.824 36.864 131.584 114.3296 131.584 203.7248 0 15.872-12.8512 28.7232-28.7232 28.7232z" fill="#F5504E" p-id="9143"></path><path d="M515.6352 72.6016c-241.0496 0-436.48 195.4304-436.48 436.48 0 73.1136 18.0736 141.9264 49.8176 202.4448 98.2528 80.7424 227.5328 123.5968 363.9808 109.1584 78.5408-8.3456 150.784-34.7648 213.0432-74.6496a28.6976 28.6976 0 0 1-9.5744-21.3504c0-93.3376-79.2576-169.216-176.64-169.216s-176.64 75.9296-176.64 169.216c0 15.872-12.8512 28.7232-28.7232 28.7232-15.872 0-28.7232-12.8512-28.7232-28.7232 0-89.3952 53.76-166.8608 131.584-203.7248-38.7072-29.5424-63.5904-75.3152-63.5904-126.6688 0-88.832 74.496-161.1264 166.0928-161.1264 91.5968 0 166.0928 72.2944 166.0928 161.1264 0 51.3536-24.9344 97.1264-63.5904 126.6688 73.984 35.0208 126.1568 106.752 131.1232 190.5152 100.0448-82.4832 165.9392-203.2128 177.408-337.1008-56.7808-175.1552-221.1328-301.7728-415.1808-301.7728z" fill="#FF5A5A" p-id="9144"></path><path d="M515.6352 72.6016c-241.0496 0-436.48 195.4304-436.48 436.48 0 25.4464 2.304 50.3296 6.5024 74.5984a489.472 489.472 0 0 0 212.4288 68.352c20.1216-57.4976 63.488-104.704 119.2448-131.1232C378.624 491.3664 353.7408 445.5936 353.7408 394.24c0-88.832 74.496-161.1264 166.0928-161.1264 91.5968 0 166.0928 72.2944 166.0928 161.1264 0 51.3536-24.9344 97.1264-63.5904 126.6688 11.0592 5.2224 21.6064 11.3152 31.6416 18.1248 98.304-82.432 162.9184-202.0864 174.2336-334.592-79.36-81.3056-190.0544-131.84-312.576-131.84z" fill="#F66A68" p-id="9145"></path><path d="M519.7824 555.4176c-70.7584 0-131.8912 40.0896-160.0512 97.7408 10.1376-0.4608 20.3264-1.1776 30.5152-2.2528 78.7968-8.3456 151.2448-34.9184 213.6576-75.0592a182.1696 182.1696 0 0 0-84.1216-20.4288z" fill="#F66A68" p-id="9146"></path><path d="M594.8928 319.5392c-19.5584-17.8688-45.9776-28.9792-75.1616-28.9792-59.904 0-108.6976 46.5408-108.6976 103.7312 0 20.3776 6.2976 39.3216 16.9472 55.3472 65.1264-31.0272 121.856-75.776 166.912-130.0992z" fill="#F67271" p-id="9147"></path><path d="M80.3328 478.208c59.2896 17.1008 122.8288 23.296 188.0064 16.384 36.7616-3.8912 72.1408-11.776 105.728-23.1424A156.02176 156.02176 0 0 1 353.6384 394.24c0-88.832 74.496-161.1264 166.0928-161.1264 41.8304 0 80.0256 15.104 109.2608 39.936a490.0608 490.0608 0 0 0 67.5328-161.1776 434.26816 434.26816 0 0 0-180.9408-39.3216c-230.656 0.0512-419.3792 179.0464-435.2512 405.6576z" fill="#F67271" p-id="9148"></path><path d="M116.7872 331.776c19.1488 0.4096 38.5024-0.3584 58.0608-2.4064 162.048-17.2032 294.7072-118.6816 359.0144-256.3072-6.0416-0.256-12.1344-0.4608-18.2272-0.4608-177.92 0-330.8544 106.496-398.848 259.1744z" fill="#F67B7A" p-id="9149"></path></svg>
 """
    return icon_string



def get_split_svg_icon():
    icon_string = """
<svg t="1744527901698" class="icon" viewBox="0 0 1175 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="7506" width="200" height="200"><path d="M1027.670251 660.645161H146.810036a146.810036 146.810036 0 0 0-146.810036 146.810036v216.544803h110.107527V807.455197a36.702509 36.702509 0 0 1 36.702509-36.702509h880.860215a36.702509 36.702509 0 0 1 36.702509 36.702509v216.544803h110.107527V807.455197a146.810036 146.810036 0 0 0-146.810036-146.810036z" fill="#4C4A58" p-id="7507"></path><path d="M1174.480287 216.544803V0h-110.107527v216.544803a36.702509 36.702509 0 0 1-36.702509 36.702509H146.810036a36.702509 36.702509 0 0 1-36.702509-36.702509V0H0v216.544803a146.810036 146.810036 0 0 0 146.810036 146.810036h880.860215a146.810036 146.810036 0 0 0 146.810036-146.810036z" fill="#4C4A58" p-id="7508"></path><path d="M178.741219 456.579211m55.053763 0l706.890323 0q55.053763 0 55.053763 55.053764l0 0q0 55.053763-55.053763 55.053763l-706.890323 0q-55.053763 0-55.053763-55.053763l0 0q0-55.053763 55.053763-55.053764Z" fill="#51C75B" p-id="7509"></path></svg>
"""
    return icon_string



def get_use_model_svg_icon():
    icon_string = """
<svg t="1744527963823" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="8582" width="200" height="200"><path d="M154.68661162 328.42831573l-1.28709316 0.64354657v364.89095508l357.81194092 188.23739678V506.36896719L154.68661162 328.42831573z" fill="#FFFFFF" p-id="8583"></path><path d="M868.37985459 328.75008945L511.21145938 140.83446552 168.52286445 321.02752959l-13.83625283 7.40078613 356.52484775 177.94065147 4.18305323-1.93064063 353.95066143-175.04469052V329.0718623l-0.96531944-0.32177285zM515.3945126 504.43832656l-4.18305323 1.93064063 4.18305323-1.93064063z" fill="#91D5FF" p-id="8584"></path><path d="M939.8135331 294.32034336a59.84983916 59.84983916 0 0 0-25.42009307-26.70718623L538.56219219 68.75723955a58.88451885 58.88451885 0 0 0-54.70146563 0L108.35125244 266.32606396a58.24097227 58.24097227 0 0 0-15.44511973 13.83625284 64.35466612 64.35466612 0 0 0-9.65320049 12.54915966 59.84983916 59.84983916 0 0 0-6.43546581 25.7418668v386.12799405a59.20629258 59.20629258 0 0 0 32.17733261 52.1272793l375.50947413 197.56882352a59.20629258 59.20629258 0 0 0 26.70718622 7.72255987h4.50482696a56.31033252 56.31033252 0 0 0 22.84590586-6.11369297l375.509475-197.56882441a59.20629258 59.20629258 0 0 0 32.17733261-52.12727842V318.45334326a56.63210625 56.63210625 0 0 0-6.4354667-24.1329999z m-70.46835908 35.07329268l-353.95066142 175.04469052-4.18305323 2.25241348v375.50947412L153.07774472 693.96281739V329.0718623l13.83625284-7.40078613 344.29746182-180.83661064 357.16839521 187.91562392z" fill="#40A9FF" p-id="8585"></path></svg>
"""
    return icon_string


def get_re_filename_svg_icon():
    icon_string = """
<svg t="1744528091835" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="9734" width="200" height="200"><path d="M938.8032 983.04H81.92V40.96h568.32l287.4368 323.9936H590.848V122.88H163.84v778.24h693.0432V482.6112h81.92z m-266.24-699.5968h82.8416L672.768 189.44z" fill="#3080E9" p-id="9735"></path><path d="M342.528 470.6304h81.92v329.8304h-81.92z" fill="#3080E9" p-id="9736"></path><path d="M195.7888 395.1616h375.5008v81.92H195.7888zM469.7088 719.0528h81.2032v81.92h-81.2032zM589.1072 719.0528h81.2032v81.92h-81.2032zM708.7104 719.0528h81.3056v81.92h-81.3056z" fill="#3080E9" p-id="9737"></path></svg>
"""
    return icon_string



def get_category_svg_icon():
    icon_string = """
<svg t="1744528172063" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="11611" width="200" height="200"><path d="M369.40572214 141.21142578H255.27236413C192.55695295 141.21142578 141.21142578 192.55695295 141.21142578 255.27236413V369.3333025c0 62.71541119 51.34552718 114.06093836 114.06093836 114.06093836H369.3333025c62.71541119 0 114.06093836-51.34552718 114.06093836-114.06093836V255.27236413C483.4666605 192.55695295 432.12113332 141.21142578 369.40572214 141.21142578z m399.32191373 0H654.59427786c-62.71541119 0-114.06093836 51.34552718-114.06093836 114.06093836V369.3333025c0 62.71541119 51.34552718 114.06093836 114.06093836 114.06093836h114.06093835c62.71541119 0 114.06093836-51.34552718 114.06093837-114.06093836V255.27236413C882.78857422 192.55695295 831.44304705 141.21142578 768.72763587 141.21142578zM369.40572214 540.5333395H255.27236413C192.55695295 540.5333395 141.21142578 591.87886668 141.21142578 654.59427786v114.06093835C141.21142578 831.44304705 192.55695295 882.78857422 255.27236413 882.78857422H369.3333025c62.71541119 0 114.06093836-51.34552718 114.06093836-114.06093836V654.59427786c0.07241965-62.71541119-51.27310752-114.06093836-113.98851872-114.06093836z m399.32191373 0H654.59427786c-62.71541119 0-114.06093836 51.34552718-114.06093836 114.06093836v114.06093835c0 62.71541119 51.34552718 114.06093836 114.06093836 114.06093837h114.06093835c62.71541119 0 114.06093836-51.34552718 114.06093837-114.06093837V654.59427786c0.07241965-62.71541119-51.27310752-114.06093836-113.98851872-114.06093836z" fill="#efabf1" p-id="11612"></path></svg>
"""
    return icon_string





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 示例 SVG 字符串

        self.setWindowTitle("SVG 字符串图标示例")
        self.setGeometry(100, 100, 800, 600)
        set_svg_icon_from_string(self, get_setting_svg_icon())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
