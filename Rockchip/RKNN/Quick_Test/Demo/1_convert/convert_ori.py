import sys
from rknn.api import RKNN

# ===== 全局常量（常量名通常用全大写） =====
# 这几个变量放在文件最上面，方便统一修改。
# Python 里没有“真正不可改”的常量，但约定全大写表示“请当作常量使用”。
DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'
DEFAULT_RKNN_PATH = '../model/yolo11.rknn'
DEFAULT_QUANT = True


# def 是“定义函数”的关键字。
# parse_arg() 表示函数名是 parse_arg，括号里是参数列表（这里没有参数）。
# 函数作用：解析命令行参数，并返回 4 个值。
def parse_arg():
    # sys.argv 是一个“列表(list)”，保存命令行参数：
    # sys.argv[0] 是脚本名，sys.argv[1] 是第一个参数，依此类推。
    # len(...) 用来获取列表长度。
    # 如果参数个数小于 3（脚本名 + 2 个必要参数），就提示用法并退出。
    if len(sys.argv) < 3:
        # str.format(...) 是字符串格式化：
        # {} 会被后面的值替换，这里把脚本名 sys.argv[0] 填进去。
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        # exit(1) 代表“异常退出”；0 通常表示正常结束，非 0 表示出错。
        exit(1)

    # 列表下标从 0 开始：
    # [1] 取第一个用户参数（onnx 模型路径）
    # [2] 取第二个用户参数（芯片平台）
    model_path = sys.argv[1]
    platform = sys.argv[2]

    # 先给 do_quant 一个默认值（来自全局常量 DEFAULT_QUANT）。
    do_quant = DEFAULT_QUANT

    # if len(sys.argv) > 3 表示用户还传了第三个参数 dtype。
    if len(sys.argv) > 3:
        model_type = sys.argv[3]

        # not in 表示“某值不在某个列表中”。
        # ['i8', 'u8', 'fp'] 是一个字符串列表。
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)

        # elif 是“否则如果”，只会在前一个 if 条件不成立时继续判断。
        # in 表示“某值在某个列表中”。
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            # 到这里说明 model_type == 'fp'，所以不做量化。
            do_quant = False

    # 处理第四个参数：输出 rknn 文件路径。
    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        # 如果没传，就使用默认输出路径。
        output_path = DEFAULT_RKNN_PATH

    # return 返回多个值时，本质上会打包成一个元组(tuple)。
    # 调用方可以一次性接收：a, b, c, d = parse_arg()
    return model_path, platform, do_quant, output_path


# __name__ 是 Python 的内置变量。
# 当这个文件被“直接运行”时，__name__ == '__main__'；
# 当这个文件被其他文件 import 时，这个条件不成立。
# 这个写法能避免“被 import 时自动执行主流程”。
if __name__ == '__main__':
    # 这里是“多变量解包赋值”：把 parse_arg() 返回的 4 个值按顺序赋给 4 个变量。
    model_path, platform, do_quant, output_path = parse_arg()

    # 创建 RKNN 对象。
    # RKNN(...) 是“调用类构造函数”来创建实例，verbose=False 表示关闭详细日志。
    rknn = RKNN(verbose=False)

    # 配置预处理。
    # 这里使用关键字参数（形如 name=value），可读性更好。
    # mean_values / std_values 用于归一化，target_platform 指定目标芯片。
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    print('done')

    # 加载 ONNX 模型。
    # ret 一般是返回码：0 表示成功，非 0 表示失败。
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        # 失败时把错误码透传出去，方便外部脚本判断失败原因。
        exit(ret)
    print('done')

    # 构建 RKNN 模型。
    # do_quantization 控制是否量化；dataset 在量化时用于校准。
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 导出 .rknn 文件到指定路径。
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 释放资源。
    # 养成习惯：对象不用时及时 release，避免资源占用。
    rknn.release()