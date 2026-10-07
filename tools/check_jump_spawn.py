"""Report URDF collision mesh heights before the first physics step."""
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'plane/resources/robots/infantry_V4/urdf/infantry_V4_long_legs_0p21_0p25.urdf'

def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, xyz[0]],
            [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, xyz[1]],
            [-sp, cp*sr, cp*cr, xyz[2]], [0, 0, 0, 1]]

def origin(element):
    if element is None:
        return transform()
    return transform(tuple(map(float, element.get('xyz', '0 0 0').split())),
                     tuple(map(float, element.get('rpy', '0 0 0').split())))

def vertices(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        for i in range(count):
            values = struct.unpack_from('<12f', data, 84 + 50*i)
            for j in (3, 6, 9):
                yield values[j:j+3]
    else:
        for line in data.decode('ascii').splitlines():
            parts = line.split()
            if parts and parts[0] == 'vertex':
                yield tuple(map(float, parts[1:4]))

def report(z, angles):
    tree = ET.parse(URDF).getroot()
    frames = {'base_link': transform((0, 0, z))}
    pending = list(tree.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in frames:
                continue
            axis = joint.find('axis')
            if joint.get('type') != 'fixed' and axis is not None:
                assert axis.get('xyz') == '0 0 1', 'Unsupported joint axis'
            local = mul(origin(joint.find('origin')), transform(rpy=(0, 0, angles.get(joint.get('name'), 0))))
            frames[joint.find('child').get('link')] = mul(frames[parent], local)
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('Disconnected URDF')
    print('Spawn z=%.3f m, terrain z=0' % z)
    for link in tree.findall('link'):
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                continue
            frame = mul(frames[link.get('name')], origin(collision.find('origin')))
            scale = tuple(map(float, mesh.get('scale', '1 1 1').split()))
            points = list(vertices((URDF.parent / mesh.get('filename')).resolve()))
            heights = [sum(frame[2][i] * v[i] * scale[i] for i in range(3)) + frame[2][3] for v in points]
            print('%-18s origin=(%.4f, %.4f, %.4f) collision z=[%.4f, %.4f]' %
                  (link.get('name'), frame[0][3], frame[1][3], frame[2][3], min(heights), max(heights)))

if __name__ == '__main__':
    angles = {'lf0_Joint': .2, 'lf1_Joint': .4, 'rf0_Joint': -.2, 'rf1_Joint': -.4}
    report(.20, angles)
    report(.30, angles)
