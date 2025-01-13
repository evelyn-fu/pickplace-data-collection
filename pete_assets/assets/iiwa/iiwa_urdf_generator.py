import lxml.etree as ET
import os
import re
import math
from pathlib import Path

class IiwaXacroProcessor:
    def __init__(self, package_path):
        self.package_path = Path(package_path)
        self.properties = {
            'PI': math.pi,
            'max_effort': 300,
            'max_velocity': 10,
            'safety_controller_k_pos': 100,
            'safety_controller_k_vel': 2,
            'joint_damping': 0.5
        }
        self.macros = {}
        self.args = {}
        
    def resolve_package_path(self, path):
        """Resolve $(find package)/rest/of/path syntax"""
        match = re.match(r'\$\(find ([^)]+)\)/(.+)', path)
        if match:
            return str(self.package_path / match.group(2))
        return path
    
    def load_xacro_file(self, filepath):
        """Load and parse a XACRO file"""
        try:
            parser = ET.XMLParser(remove_blank_text=True, remove_comments=True)
            tree = ET.parse(filepath, parser)
            return tree.getroot()
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def evaluate_expression(self, expr):
        """Evaluate a mathematical expression with property substitution"""
        if expr is None:
            return expr
        
        # First handle $(arg ...) substitution
        def replace_arg(match):
            arg_name = match.group(1)
            if arg_name in self.args:
                return str(self.args[arg_name])
            print(f"Warning: Argument '{arg_name}' not found")
            return match.group(0)
        
        expr = re.sub(r'\$\(arg ([^)]+)\)', replace_arg, str(expr))
        
        # Then handle ${...} substitution
        def replace_var(match):
            var_name = match.group(1).strip()
            # First check if it's a property or argument
            if var_name in self.properties:
                return str(self.properties[var_name])
            elif var_name in self.args:
                return str(self.args[var_name])
            # If not found as property/arg, try evaluating as mathematical expression
            try:
                safe_dict = {
                    'PI': math.pi,
                    'pi': math.pi,
                    'sin': math.sin,
                    'cos': math.cos,
                    'tan': math.tan,
                    'asin': math.asin,
                    'acos': math.acos,
                    'atan': math.atan,
                    'atan2': math.atan2,
                    'abs': abs,
                    'pow': pow
                }
                result = eval(var_name, {"__builtins__": {}}, safe_dict)
                if isinstance(result, float):
                    return f"{result:.6f}".rstrip('0').rstrip('.')
                return str(result)
            except Exception as e:
                print(f"Warning: Neither property nor valid expression '{var_name}' found")
                return match.group(0)
        
        expr = re.sub(r'\$\{([^}]+)\}', replace_var, expr)
        
        # If the expression is now just a number or string, return it
        if not any(c in expr for c in '+-*/()'):
            return expr
            
        # Evaluate any remaining mathematical expression
        try:
            safe_dict = {
                'PI': math.pi,
                'pi': math.pi,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'atan2': math.atan2,
                'abs': abs,
                'pow': pow
            }
            
            result = eval(expr, {"__builtins__": {}}, safe_dict)
            
            if isinstance(result, float):
                return f"{result:.6f}".rstrip('0').rstrip('.')
            return str(result)
        except Exception as e:
            print(f"Error evaluating expression '{expr}': {e}")
            return expr
    
    def get_tag_without_ns(self, element):
        """Get tag name without namespace"""
        if isinstance(element, ET._Comment):
            return None
        tag = str(element.tag)
        return tag.split('}')[-1] if '}' in tag else tag
    
    def process_include(self, element, current_dir):
        """Process xacro:include tags"""
        filename = element.get('filename')
        if filename:
            resolved_path = self.resolve_package_path(filename)
            print(f"Including file: {resolved_path}")
            included_root = self.load_xacro_file(resolved_path)
            if included_root is not None:
                self.process_element(included_root, os.path.dirname(resolved_path))
                return included_root
        return None
    
    def process_property(self, element):
        """Process xacro:property tags"""
        name = element.get('name')
        value = element.get('value')
        if name and value:
            evaluated_value = self.evaluate_expression(value)
            self.properties[name] = evaluated_value
            print(f"Property defined: {name} = {evaluated_value}")
    
    def process_arg(self, element):
        """Process xacro:arg tags"""
        name = element.get('name')
        default = element.get('default')
        if name:
            if name not in self.args:  # Only set if not already defined
                evaluated_default = self.evaluate_expression(default) if default else default
                self.args[name] = evaluated_default
                print(f"Argument defined: {name} = {evaluated_default}")
    
    def substitute_expressions(self, text):
        """Substitute ${expressions} and $(arg ...) with their evaluated values"""
        if text is None:
            return text
        
        # First substitute $(arg ...) expressions
        def replace_arg(match):
            arg_name = match.group(1)
            if arg_name in self.args:
                return str(self.args[arg_name])
            print(f"Warning: Argument '{arg_name}' not found")
            return match.group(0)
        
        text = re.sub(r'\$\(arg ([^)]+)\)', replace_arg, str(text))
        
        # Then substitute ${...} expressions
        def replace_expr(match):
            expr = match.group(1)
            return self.evaluate_expression('${'+expr+'}')
        
        return re.sub(r'\$\{([^}]+)\}', replace_expr, text)
    
    def process_element(self, element, current_dir):
        """Process a XACRO element and its children"""
        if isinstance(element, ET._Comment):
            return
            
        tag = self.get_tag_without_ns(element)
        
        # Handle special tags
        if tag == 'include':
            included = self.process_include(element, current_dir)
            if included is not None:
                parent = element.getparent()
                if parent is not None:
                    idx = list(parent).index(element)
                    parent.remove(element)
                    for child in included:
                        parent.insert(idx, child)
                        idx += 1
        elif tag == 'property':
            self.process_property(element)
            element.getparent().remove(element)
        elif tag == 'arg':
            self.process_arg(element)
            element.getparent().remove(element)
            return
            
        # Process attributes
        for key, value in list(element.attrib.items()):
            element.attrib[key] = self.substitute_expressions(value)
            
        # Process child elements
        for child in list(element):
            self.process_element(child, current_dir)
    
    def convert(self, input_file, output_file):
        """Convert XACRO file to URDF"""
        ET.register_namespace('xacro', "http://www.ros.org/wiki/xacro")
        print(f"Processing input file: {input_file}")
        
        root = self.load_xacro_file(input_file)
        if root is None:
            return False
        
        # Process all elements
        self.process_element(root, os.path.dirname(input_file))
        
        # Clean up remaining xacro elements and namespaces
        for element in root.iter():
            if isinstance(element, ET._Comment):
                continue
            tag = self.get_tag_without_ns(element)
            if tag and tag != element.tag:
                element.tag = tag
        
        # undo the the iiwa7 macro to reveal the urdf, detete the first joint with the parent substitutions
        # and then delete all other macros
        for macro_el in root.findall('.//macro'):
            if not macro_el.attrib['name'] == 'iiwa7':
                parent = macro_el.getparent()
                parent.remove(macro_el)
            else:
                children = macro_el.getchildren() 
                parent = macro_el.getparent()
                parent.remove(macro_el) 
                for c in children:
                    if c.tag == 'iiwa_gazebo':
                        continue
                    if c.tag == 'iiwa_transmission':
                        continue
                    if not 'name' in c.keys():
                        parent.append(c)
                        continue
                    if c.attrib['name']!= "${parent}_iiwa_joint":
                        parent.append(c)


        for n in ['self_collision_checking', 'iiwa7']:
            for el in root.findall(f'.//{n}'):
                parent = el.getparent()
                parent.remove(el)

        for el in root.findall('.//link'):
            if el.attrib['name'] =='world':
                parent = el.getparent()
                parent.remove(el)

        # Write the processed XML to file
        tree = ET.ElementTree(root)
        tree.write(output_file, 
                  xml_declaration=True, 
                  encoding='utf-8', 
                  pretty_print=True)
        print(f"Written output to: {output_file}")
        return True

def convert_iiwa_xacro(package_path, input_file, output_file):
    processor = IiwaXacroProcessor(package_path)
    # Set the default args that match your root xacro
    processor.args = {
        'hardware_interface': 'PositionJointInterface',
        'robot_name': 'iiwa',
        'origin_xyz': '0 0 0',  
        'origin_rpy': '0 0 0'
    }
    success = processor.convert(input_file, output_file)
    if success:
        print(f"Successfully converted {input_file} to {output_file}")
    else:
        print(f"Failed to convert {input_file}")

if __name__ == "__main__":
    package_path = "assets/iiwa"
    input_file = "assets/iiwa/urdf/iiwa7.urdf.xacro"
    output_file = "assets/iiwa/urdf_output/iiwa7.urdf"
    
    convert_iiwa_xacro(package_path, input_file, output_file)