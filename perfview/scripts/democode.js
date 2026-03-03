class Model{
  constructor(){
    this.q1 = new nn.QuantizeLinear();
    this.dq1 = new nn.DequantizeLinear();
    this.q2 = new nn.QuantizeLinear(axis=0);
    this.dq2 = new nn.DequantizeLinear(axis=0);
    this.conv1 = new nn.Conv(kernel=1, padding=0, stride=1, dilation=1, group=1);
    this.bn1 = new nn.BatchNormalization(epsilon=0.000009999999747378752, momentum=0.8999999761581421);
    this.relu1 = new nn.Relu();
    this.q3 = new nn.QuantizeLinear();
    this.dq3 = new nn.DequantizeLinear();
    this.q4 = new nn.QuantizeLinear(axis=0);
    this.dq4 = new nn.DequantizeLinear(axis=0);
    this.conv2 = new nn.Conv(kernel=1, padding=0, stride=1, dilation=1, group=1);
  }

  init_weights(){
    this.c1 = nn.load("/kep_head/cls_block/conv/_input_quantizer/Constant_1_output_0");
    this.c2 = nn.load("/kep_head/cls_block/conv/_input_quantizer/Constant_output_0");
    this.c3 = nn.load("kep_head.cls_block.conv.weight");
    this.c4 = nn.load("/kep_head/cls_block/conv/_weight_quantizer/Constant_output_0");
    this.c5 = nn.load("/kep_head/cls_block/conv/_weight_quantizer/Constant_1_output_0");
    this.c6 = nn.load("kep_head.cls_block.bn.weight");
    this.c7 = nn.load("kep_head.cls_block.bn.bias");
    this.c8 = nn.load("kep_head.cls_block.bn.running_mean");
    this.c9 = nn.load("kep_head.cls_block.bn.running_var");
    this.c10 = nn.load("/kep_head/cls_pred/_input_quantizer/Constant_1_output_0");
    this.c11 = nn.load("/kep_head/cls_pred/_input_quantizer/Constant_output_0");
    this.c12 = nn.load("kep_head.cls_pred.weight");
    this.c13 = nn.load("/kep_head/cls_pred/_weight_quantizer/Constant_output_0");
    this.c14 = nn.load("/kep_head/cls_pred/_weight_quantizer/Constant_1_output_0");
    this.c15 = nn.load("kep_head.cls_pred.bias");
  }

  forward(input1){
    x1 = this.q1(input1, this.c1, this.c2);
    x2 = this.dq1(x1, this.c1, this.c2);
    x3 = this.q2(this.c3, this.c4, this.c5);
    x4 = this.dq2(x3, this.c4, this.c5);
    x5 = this.conv1(x2, x4);
    x6 = this.bn1(x5, this.c6, this.c7, this.c8, this.c9);
    x7 = this.relu1(x6);
    x8 = this.q3(x7, this.c10, this.c11);
    x9 = this.dq3(x8, this.c10, this.c11);
    x10 = this.q4(this.c12, this.c13, this.c14);
    x11 = this.dq4(x10, this.c13, this.c14);
    output1 = this.conv2(x9, x11, this.c15);
    return output1;
  }
}
