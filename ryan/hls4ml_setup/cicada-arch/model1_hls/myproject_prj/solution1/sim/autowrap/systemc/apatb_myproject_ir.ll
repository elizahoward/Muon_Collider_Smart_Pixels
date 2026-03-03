; ModuleID = '/home/ryanmichaud/common_repo/Muon_Collider_Smart_Pixels/ryan/hls4ml_setup/cicada-arch/model1_hls/myproject_prj/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>" = type { %"struct.ap_fixed_base<16, 6, true, AP_TRN, AP_WRAP, 0>" }
%"struct.ap_fixed_base<16, 6, true, AP_TRN, AP_WRAP, 0>" = type { %"struct.ssdm_int<16, true>" }
%"struct.ssdm_int<16, true>" = type { i16 }

; Function Attrs: inaccessiblemem_or_argmemonly noinline willreturn
define void @apatb_myproject_ir(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %z_global, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %x_size, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %y_size, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="1" %y_local, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="1" "partition" %layer13_out) local_unnamed_addr #0 {
entry:
  %z_global_copy7 = alloca i16, align 512
  %x_size_copy8 = alloca i16, align 512
  %y_size_copy9 = alloca i16, align 512
  %y_local_copy10 = alloca i16, align 512
  %layer13_out_copy6 = alloca i16, align 512
  %0 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %z_global to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %1 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %x_size to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %2 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %y_size to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %3 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %y_local to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  %4 = bitcast %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %layer13_out to [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]*
  call void @copy_in([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %0, i16* nonnull align 512 %z_global_copy7, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %1, i16* nonnull align 512 %x_size_copy8, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %2, i16* nonnull align 512 %y_size_copy9, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %3, i16* nonnull align 512 %y_local_copy10, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %4, i16* nonnull align 512 %layer13_out_copy6)
  call void @apatb_myproject_hw(i16* %z_global_copy7, i16* %x_size_copy8, i16* %y_size_copy9, i16* %y_local_copy10, i16* %layer13_out_copy6)
  call void @copy_back([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i16* %z_global_copy7, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %1, i16* %x_size_copy8, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2, i16* %y_size_copy9, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %3, i16* %y_local_copy10, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %4, i16* %layer13_out_copy6)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.65.114.115"(i16* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = load i16, i16* %src.addr.0.0.05, align 2
  store i16 %1, i16* %dst, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.113.116"(i16* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.65.114.115"(i16* %dst, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.58.124.125"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i16* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr.0.0.06 = getelementptr [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = load i16, i16* %src, align 2
  store i16 %1, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.55.123.126"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i16* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.58.124.125"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i16* %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.65"(i16* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, i64 %dst_shift, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  %1 = trunc i64 %dst_shift to i16
  %2 = shl i16 -1, %1
  %3 = xor i16 %2, -1
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %4 = load i16, i16* %src.addr.0.0.05, align 2
  %5 = load i16, i16* %dst, align 2
  %6 = shl i16 %4, %1
  %7 = and i16 %5, %3
  %8 = or i16 %7, %6
  store i16 %8, i16* %dst, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i16* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.65"(i16* %dst, i64 0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_in([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="0" "unpacked"="0", i16* noalias nocapture align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture align 512 "orig.arg.no"="3" "unpacked"="3.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="4" "unpacked"="4", i16* noalias nocapture align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="8" "unpacked"="8", i16* noalias nocapture align 512 "orig.arg.no"="9" "unpacked"="9.0") #3 {
entry:
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i16* align 512 %1, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i16* align 512 %3, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i16* align 512 %5, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %4)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"(i16* align 512 %7, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.113.116"(i16* align 512 %9, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.58"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i16* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 %src_shift, i64 "orig.arg.no"="2" "unpacked"="2" %num) #1 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  %1 = trunc i64 %src_shift to i16
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr.0.0.06 = getelementptr [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"], [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %2 = load i16, i16* %src, align 2
  %3 = lshr i16 %2, %1
  store i16 %3, i16* %dst.addr.0.0.06, align 2
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.55"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i16* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #2 {
entry:
  %0 = icmp eq [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.58"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i16* %src, i64 0, i64 1)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_out([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i16* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="4" "unpacked"="4", i16* noalias nocapture readonly align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture readonly align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="8" "unpacked"="8", i16* noalias nocapture readonly align 512 "orig.arg.no"="9" "unpacked"="9.0") #4 {
entry:
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.55"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %0, i16* align 512 %1)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.55"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %2, i16* align 512 %3)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.55"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %4, i16* align 512 %5)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.55"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6, i16* align 512 %7)
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.55.123.126"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8, i16* align 512 %9)
  ret void
}

declare void @apatb_myproject_hw(i16*, i16*, i16*, i16*, i16*)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_back([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i16* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i16* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="4" "unpacked"="4", i16* noalias nocapture readonly align 512 "orig.arg.no"="5" "unpacked"="5.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="6" "unpacked"="6", i16* noalias nocapture readonly align 512 "orig.arg.no"="7" "unpacked"="7.0", [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="8" "unpacked"="8", i16* noalias nocapture readonly align 512 "orig.arg.no"="9" "unpacked"="9.0") #4 {
entry:
  call void @"onebyonecpy_hls.p0a1struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>.55.123.126"([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8, i16* align 512 %9)
  ret void
}

define void @myproject_hw_stub_wrapper(i16*, i16*, i16*, i16*, i16*) #5 {
entry:
  %5 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %6 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %7 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %8 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  %9 = alloca [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]
  call void @copy_out([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %5, i16* %0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6, i16* %1, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %7, i16* %2, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8, i16* %3, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %9, i16* %4)
  %10 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %5 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %11 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %12 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %7 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %13 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  %14 = bitcast [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %9 to %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"*
  call void @myproject_hw_stub(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %10, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %11, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %12, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %13, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* %14)
  call void @copy_in([1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %5, i16* %0, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %6, i16* %1, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %7, i16* %2, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %8, i16* %3, [1 x %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"]* %9, i16* %4)
  ret void
}

declare void @myproject_hw_stub(%"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<16, 6, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull)

attributes #0 = { inaccessiblemem_or_argmemonly noinline willreturn "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="arraycpy_hls" }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #4 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #5 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}
!datalayout.transforms.on.top = !{!5}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
!5 = !{!6, !8, !10}
!6 = !{!7}
!7 = !{!"4.0", [1 x i16]* null}
!8 = !{!9}
!9 = !{!"array_partition", !"type=Complete", !"dim=1"}
!10 = !{!11}
!11 = !{!"4.0", i16* null}
