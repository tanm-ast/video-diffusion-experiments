"""
Generic Forward Pass Tracer

A reusable utility to trace tensor shapes, execution order, and module
behavior through any PyTorch model.

Usage:
    from forward_tracer import ForwardTracer
    
    tracer = ForwardTracer(model)
    output = tracer.trace(input_tensor)
    tracer.print_summary()
    tracer.save_report("trace.txt")
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ModuleTrace:
    """Record of a single module's forward pass."""
    name: str
    class_name: str
    input_shapes: List[Tuple]
    output_shapes: List[Tuple]
    input_dtypes: List[str]
    output_dtypes: List[str]
    num_parameters: int
    execution_order: int
    depth: int  # nesting level in module hierarchy
    extra_info: Dict = field(default_factory=dict)


class ForwardTracer:
    """
    Instruments a PyTorch model to trace the forward pass.
    
    Example:
        model = SomeModel()
        tracer = ForwardTracer(model)
        
        # Run traced forward pass
        output = tracer.trace(input_tensor)
        
        # Analyze results
        tracer.print_summary()
        tracer.print_execution_order()
        tracer.save_report("trace_report.txt")
        
        # Clean up
        tracer.remove_hooks()
    """
    
    def __init__(
        self,
        model: nn.Module,
        trace_depth: Optional[int] = None,
        module_filter: Optional[callable] = None,
        capture_tensors: bool = False
    ):
        """
        Args:
            model: The PyTorch model to trace
            trace_depth: Max depth to trace (None = all)
            module_filter: Function(name, module) -> bool to select modules
            capture_tensors: If True, store actual tensors (memory intensive!)
        """
        self.model = model
        self.trace_depth = trace_depth
        self.module_filter = module_filter or (lambda n, m: True)
        self.capture_tensors = capture_tensors
        
        self.traces: OrderedDict[str, ModuleTrace] = OrderedDict()
        self.execution_order: List[str] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.captured_tensors: Dict[str, Dict] = {}
        
        self._execution_counter = 0
        self._register_hooks()
    
    def _get_depth(self, name: str) -> int:
        """Calculate module depth from name."""
        return name.count('.') if name else 0
    
    def _get_shapes(self, tensor_or_tuple) -> List[Tuple]:
        """Extract shapes from tensor or tuple of tensors."""
        if tensor_or_tuple is None:
            return [()]
        if isinstance(tensor_or_tuple, torch.Tensor):
            return [tuple(tensor_or_tuple.shape)]
        if isinstance(tensor_or_tuple, (tuple, list)):
            shapes = []
            for t in tensor_or_tuple:
                if isinstance(t, torch.Tensor):
                    shapes.append(tuple(t.shape))
                elif t is None:
                    shapes.append(())
                else:
                    shapes.append(("non-tensor", type(t).__name__))
            return shapes
        return [("non-tensor", type(tensor_or_tuple).__name__)]
    
    def _get_dtypes(self, tensor_or_tuple) -> List[str]:
        """Extract dtypes from tensor or tuple of tensors."""
        if tensor_or_tuple is None:
            return ["None"]
        if isinstance(tensor_or_tuple, torch.Tensor):
            return [str(tensor_or_tuple.dtype)]
        if isinstance(tensor_or_tuple, (tuple, list)):
            dtypes = []
            for t in tensor_or_tuple:
                if isinstance(t, torch.Tensor):
                    dtypes.append(str(t.dtype))
                else:
                    dtypes.append(type(t).__name__)
            return dtypes
        return [type(tensor_or_tuple).__name__]
    
    def _make_hook(self, name: str, depth: int):
        """Create a forward hook for a specific module."""
        def hook(module, input, output):
            # Extract shapes and dtypes
            input_shapes = self._get_shapes(input)
            output_shapes = self._get_shapes(output)
            input_dtypes = self._get_dtypes(input)
            output_dtypes = self._get_dtypes(output)
            
            # Count parameters
            num_params = sum(p.numel() for p in module.parameters(recurse=False))
            
            # Build extra info based on module type
            extra = {}
            if hasattr(module, 'in_features'):
                extra['in_features'] = module.in_features
            if hasattr(module, 'out_features'):
                extra['out_features'] = module.out_features
            if hasattr(module, 'kernel_size'):
                extra['kernel_size'] = module.kernel_size
            if hasattr(module, 'heads'):
                extra['heads'] = module.heads
            if hasattr(module, 'num_heads'):
                extra['num_heads'] = module.num_heads
            
            # Create trace record
            trace = ModuleTrace(
                name=name,
                class_name=module.__class__.__name__,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                input_dtypes=input_dtypes,
                output_dtypes=output_dtypes,
                num_parameters=num_params,
                execution_order=self._execution_counter,
                depth=depth,
                extra_info=extra
            )
            
            self.traces[name] = trace
            self.execution_order.append(name)
            self._execution_counter += 1
            
            # Optionally capture tensors
            if self.capture_tensors:
                self.captured_tensors[name] = {
                    'input': input[0].detach().cpu() if isinstance(input[0], torch.Tensor) else None,
                    'output': output.detach().cpu() if isinstance(output, torch.Tensor) else None
                }
        
        return hook
    
    def _register_hooks(self):
        """Register forward hooks on all matching modules."""
        for name, module in self.model.named_modules():
            if not name:  # Skip root
                continue
            
            depth = self._get_depth(name)
            
            # Apply depth filter
            if self.trace_depth is not None and depth > self.trace_depth:
                continue
            
            # Apply custom filter
            if not self.module_filter(name, module):
                continue
            
            hook = module.register_forward_hook(self._make_hook(name, depth))
            self.hooks.append(hook)
    
    def trace(self, *args, **kwargs):
        """Run forward pass and collect traces."""
        self.traces.clear()
        self.execution_order.clear()
        self.captured_tensors.clear()
        self._execution_counter = 0
        
        with torch.no_grad():
            output = self.model(*args, **kwargs)
        
        return output
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def print_summary(self, max_name_width: int = 60):
        """Print a summary table of all traced modules."""
        if not self.traces:
            print("No traces collected. Run trace() first.")
            return
        
        print("\n" + "=" * 100)
        print("FORWARD PASS SUMMARY")
        print("=" * 100)
        
        header = f"{'Order':<6} {'Module Name':<{max_name_width}} {'Type':<25} {'Input':<20} {'Output':<20}"
        print(header)
        print("-" * 100)
        
        for name in self.execution_order:
            trace = self.traces[name]
            indent = "  " * trace.depth
            short_name = indent + name.split('.')[-1]
            if len(short_name) > max_name_width:
                short_name = short_name[:max_name_width-3] + "..."
            
            in_shape = str(trace.input_shapes[0]) if trace.input_shapes else "?"
            out_shape = str(trace.output_shapes[0]) if trace.output_shapes else "?"
            
            if len(in_shape) > 18:
                in_shape = in_shape[:15] + "..."
            if len(out_shape) > 18:
                out_shape = out_shape[:15] + "..."
            
            print(f"{trace.execution_order:<6} {short_name:<{max_name_width}} {trace.class_name:<25} {in_shape:<20} {out_shape:<20}")
    
    def print_execution_order(self):
        """Print just the execution order with full names."""
        print("\n" + "=" * 60)
        print("EXECUTION ORDER")
        print("=" * 60)
        
        for i, name in enumerate(self.execution_order):
            trace = self.traces[name]
            print(f"{i:4d}. {name} ({trace.class_name})")
    
    def print_module_details(self, module_name: str):
        """Print detailed information about a specific module."""
        if module_name not in self.traces:
            print(f"Module '{module_name}' not found in traces.")
            return
        
        trace = self.traces[module_name]
        
        print(f"\n{'=' * 60}")
        print(f"MODULE: {module_name}")
        print(f"{'=' * 60}")
        print(f"Type:            {trace.class_name}")
        print(f"Execution order: {trace.execution_order}")
        print(f"Depth:           {trace.depth}")
        print(f"Parameters:      {trace.num_parameters:,}")
        print(f"\nInput shapes:    {trace.input_shapes}")
        print(f"Input dtypes:    {trace.input_dtypes}")
        print(f"Output shapes:   {trace.output_shapes}")
        print(f"Output dtypes:   {trace.output_dtypes}")
        
        if trace.extra_info:
            print(f"\nExtra info:")
            for k, v in trace.extra_info.items():
                print(f"  {k}: {v}")
    
    def get_modules_by_type(self, class_name: str) -> List[ModuleTrace]:
        """Get all traces for modules of a specific type."""
        return [t for t in self.traces.values() if t.class_name == class_name]
    
    def find_shape_changes(self) -> List[Tuple[str, Tuple, Tuple]]:
        """Find all modules where input shape != output shape."""
        changes = []
        for name, trace in self.traces.items():
            if trace.input_shapes and trace.output_shapes:
                if trace.input_shapes[0] != trace.output_shapes[0]:
                    changes.append((name, trace.input_shapes[0], trace.output_shapes[0]))
        return changes
    
    def save_report(self, filepath: str):
        """Save a detailed report to a text file."""
        with open(filepath, 'w') as f:
            f.write(f"Forward Pass Trace Report\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model.__class__.__name__}\n")
            f.write(f"Total modules traced: {len(self.traces)}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXECUTION ORDER:\n")
            f.write("-" * 80 + "\n")
            for i, name in enumerate(self.execution_order):
                trace = self.traces[name]
                f.write(f"{i:4d}. [{trace.class_name}] {name}\n")
                f.write(f"       Input:  {trace.input_shapes} {trace.input_dtypes}\n")
                f.write(f"       Output: {trace.output_shapes} {trace.output_dtypes}\n")
                if trace.num_parameters > 0:
                    f.write(f"       Params: {trace.num_parameters:,}\n")
                f.write("\n")
            
            f.write("\nSHAPE CHANGES:\n")
            f.write("-" * 80 + "\n")
            for name, in_shape, out_shape in self.find_shape_changes():
                f.write(f"{name}: {in_shape} -> {out_shape}\n")
    
    def to_dict(self) -> Dict:
        """Export traces as a dictionary (for JSON serialization)."""
        return {
            'model_class': self.model.__class__.__name__,
            'execution_order': self.execution_order,
            'traces': {
                name: {
                    'class_name': t.class_name,
                    'input_shapes': [list(s) if isinstance(s, tuple) else s for s in t.input_shapes],
                    'output_shapes': [list(s) if isinstance(s, tuple) else s for s in t.output_shapes],
                    'num_parameters': t.num_parameters,
                    'execution_order': t.execution_order,
                    'extra_info': t.extra_info
                }
                for name, t in self.traces.items()
            }
        }


# =============================================================================
# Convenience functions
# =============================================================================

def trace_model(model: nn.Module, sample_input: torch.Tensor, **kwargs) -> ForwardTracer:
    """
    Quick function to trace a model with a sample input.
    
    Example:
        tracer = trace_model(my_model, torch.randn(1, 3, 224, 224))
        tracer.print_summary()
    """
    tracer = ForwardTracer(model, **kwargs)
    tracer.trace(sample_input)
    return tracer


def print_model_flow(model: nn.Module, sample_input: torch.Tensor):
    """
    Quick function to print the forward pass flow.
    
    Example:
        print_model_flow(my_model, torch.randn(1, 3, 224, 224))
    """
    tracer = trace_model(model, sample_input)
    tracer.print_summary()
    tracer.remove_hooks()


# =============================================================================
# Specialized tracers
# =============================================================================

class AttentionTracer(ForwardTracer):
    """
    Specialized tracer for attention-based models.
    Captures attention weights when available.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        # Filter for attention-related modules
        def attention_filter(name, module):
            class_name = module.__class__.__name__.lower()
            name_lower = name.lower()
            return 'attention' in class_name or 'attn' in name_lower
        
        super().__init__(model, module_filter=attention_filter, **kwargs)


# =============================================================================
# Demo / Test
# =============================================================================

if __name__ == "__main__":
    # Demo with a simple model
    print("Testing ForwardTracer with a simple model...\n")
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool(self.conv2(x))
            x = x.flatten(1)
            x = self.fc(x)
            return x
    
    model = DemoModel()
    sample = torch.randn(2, 3, 32, 32)
    
    tracer = ForwardTracer(model)
    output = tracer.trace(sample)
    
    tracer.print_summary()
    tracer.print_execution_order()
    
    print("\nShape changes:")
    for name, in_s, out_s in tracer.find_shape_changes():
        print(f"  {name}: {in_s} -> {out_s}")
    
    tracer.remove_hooks()
    print("\nDone!")
