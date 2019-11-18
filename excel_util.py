import xlwt


def set_excel_style(font_height=200, bold=False, alignment_wrap=False):
    """
    设置excel打印的格式
    :param font_height: 字体大小
    :param bold:    是否加粗
    :return:
    """
    style = xlwt.XFStyle()  # 初始化样式
    font = xlwt.Font()  # 为样式创建字体
    font.name = 'Times New Roman'
    font.bold = bold  # 黑体
    font.height = font_height
    alignment = xlwt.Alignment()
    if alignment_wrap:
        alignment.horz = xlwt.Alignment.HORZ_CENTER  # 设置水平居中
        alignment.vert = xlwt.Alignment.VERT_CENTER  # 设置垂直居中
    else:
        alignment.horz = xlwt.Alignment.HORZ_CENTER  # 设置水平居中
        alignment.vert = xlwt.Alignment.VERT_CENTER  # 设置垂直居中
        alignment.wrap = xlwt.Alignment.WRAP_AT_RIGHT  # 自动换行
    style.alignment = alignment
    style.font = font  # 设定样式

    return style