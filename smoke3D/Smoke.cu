#define _USE_MATH_DEFINES
#include <math.h>

#include "constants.h"
#include "Smoke.h"

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <algorithm>

#include "helper_cuda.h"

#define FOR_EACH_CELL  \
    for (int k = 0; k < zRes; ++k) \
        for (int j = 0; j < yRes; ++j) \
            for (int i = 0; i < xRes; ++i)

#define FOR_EACH_FACE_X  \
    for (int k = 0; k < zRes; ++k) \
        for (int j = 0; j < yRes; ++j) \
            for (int i = 0; i < xRes + 1; ++i)

#define FOR_EACH_FACE_Y  \
    for (int k = 0; k < zRes; ++k) \
        for (int j = 0; j < yRes + 1; ++j) \
            for (int i = 0; i < xRes; ++i)

#define FOR_EACH_FACE_Z  \
    for (int k = 0; k < zRes + 1; ++k) \
        for (int j = 0; j < yRes; ++j) \
            for (int i = 0; i < xRes; ++i)

Smoke::Smoke()
    : dt(0.0f), t(0.0f), next_shutter_time(0.0f), isTimeToRender(false)
{
    //dt = std::min(CFL * DX / INIT_VELOCITY, 1.0f / FPS);
    dt = DT;

    if (SAVE_VDB)
    {
        openvdb::initialize();
    }
}

Smoke::~Smoke()
{
}

void Smoke::initialize()
{
    // initial setup
    printf("initializing data...");
    initVelocity();
    initDensity();
    initTemperature();
    setObstacles();

    d_force_y_data.resize(xRes * yRes * zRes);
    d_pressure_data.resize(xRes * yRes * zRes);
    d_Ax_data.resize(xRes * yRes * zRes);
    d_divergence_data.resize(xRes * yRes * zRes);
    d_direction_data.resize(xRes * yRes * zRes);

    force_y.data = d_force_y_data.data();
    pressure.data = d_pressure_data.data();
    Ax.data = d_Ax_data.data();
    divergence.data = d_divergence_data.data();
    direction.data = d_direction_data.data();

    d_speed.resize(blockArraySize);

    printf("Done\n");
}

void Smoke::setNextShutterTime()
{
    next_shutter_time = t + 1.0f / FPS;
}


void Smoke::addScalarFieldForVDB(openvdb::GridPtrVec& grids, std::string name, const ScalarField& field) const
{
    openvdb::FloatGrid::Ptr scalarGrid = openvdb::FloatGrid::create(0.0f);
    scalarGrid->setName(name);
    scalarGrid->setTransform(openvdb::math::Transform::createLinearTransform(DX));
    scalarGrid->setGridClass(openvdb::GRID_FOG_VOLUME);

    auto scalarAccessor = scalarGrid->getAccessor();

    thrust::host_vector<float> h_data(xRes * yRes * zRes);

    thrust::copy_n(field.data, xRes * yRes * zRes, h_data.data());

    FOR_EACH_CELL
    {
        int offset = i + (j + k * yRes) * xRes;
        openvdb::Coord xyz(i, yRes - 1 - j, k);
        float value = h_data[offset];
        scalarAccessor.setValue(xyz, value);
    }

    grids.push_back(scalarGrid);
}

void Smoke::addUFieldForVDB(openvdb::GridPtrVec& grids, const uField& field) const
{
    openvdb::FloatGrid::Ptr staggeredGrid = openvdb::FloatGrid::create(0.0);
    staggeredGrid->setName("vel.x");
    staggeredGrid->setTransform(openvdb::math::Transform::createLinearTransform(DX));
    staggeredGrid->setGridClass(openvdb::GRID_STAGGERED);

    auto velAccessor = staggeredGrid->getAccessor();

    thrust::host_vector<float> h_data((xRes + 1) * yRes * zRes);

    thrust::copy_n(field.data, (xRes + 1) * yRes * zRes, h_data.data());

    FOR_EACH_FACE_X
    {
        int offset = i + (j + k * yRes) * (xRes + 1);
        openvdb::Coord xyz(i, yRes - 1 - j, k);
        float value = h_data[offset];
        velAccessor.setValue(xyz, value);
    }

    grids.push_back(staggeredGrid);

}

void Smoke::addVFieldForVDB(openvdb::GridPtrVec& grids, const vField& field) const
{
    openvdb::FloatGrid::Ptr staggeredGrid = openvdb::FloatGrid::create(0.0);
    staggeredGrid->setName("vel.y");
    staggeredGrid->setTransform(openvdb::math::Transform::createLinearTransform(DX));
    staggeredGrid->setGridClass(openvdb::GRID_STAGGERED);

    auto velAccessor = staggeredGrid->getAccessor();

    thrust::host_vector<float> h_data(xRes * (yRes + 1) * zRes);

    thrust::copy_n(field.data, xRes * (yRes + 1) * zRes, h_data.data());

    FOR_EACH_FACE_Y
    {
        int offset = i + (j + k * (yRes+1)) * xRes;
        openvdb::Coord xyz(i, yRes - j, k);
        float value = h_data[offset];
        velAccessor.setValue(xyz, value);
    }

    grids.push_back(staggeredGrid);
}

void Smoke::addWFieldForVDB(openvdb::GridPtrVec& grids, const wField& field) const
{
    openvdb::FloatGrid::Ptr staggeredGrid = openvdb::FloatGrid::create(0.0);
    staggeredGrid->setName("vel.z");
    staggeredGrid->setTransform(openvdb::math::Transform::createLinearTransform(DX));
    staggeredGrid->setGridClass(openvdb::GRID_STAGGERED);

    auto velAccessor = staggeredGrid->getAccessor();

    thrust::host_vector<float> h_data(xRes * yRes * (zRes + 1));

    thrust::copy_n(field.data, xRes * yRes * (zRes + 1), h_data.data());

    FOR_EACH_FACE_Z
    {
        int offset = i + (j + k * yRes) * xRes;
        openvdb::Coord xyz(i, yRes - 1 - j, k);
        float value = h_data[offset];
        velAccessor.setValue(xyz, value);
    }

    grids.push_back(staggeredGrid);
}

void Smoke::saveVDB() const
{
    static int count = 0;
    std::string filename;
    std::stringstream ss;
    ss << "VDB/res" << xRes << "x" << yRes << "x" << zRes << "_" << std::setfill('0') << std::setw(4) << std::right << std::to_string(count++) << ".vdb";
    ss >> filename;

    openvdb::GridPtrVec grids;

    addScalarFieldForVDB(grids, "density", density0);
    addScalarFieldForVDB(grids, "temperature", temperature0);
    addScalarFieldForVDB(grids, "pressure", pressure);
    addUFieldForVDB(grids, u0);
    addVFieldForVDB(grids, v0);
    addWFieldForVDB(grids, w0);

    // Create a VDB file object.
    openvdb::io::File file(filename);

    // Write out the contents of the container.
    file.write(grids);
    file.close();
}


void Smoke::initVelocity()
{
    thrust::host_vector<float> h_u0_scanline((xRes + 1) * yRes * zRes);
    thrust::host_vector<float> h_v0_scanline(xRes * (yRes + 1) * zRes);
    thrust::host_vector<float> h_w0_scanline(xRes * yRes * (zRes + 1));

    d_u0_data.resize((xRes + 1) * yRes * zRes);
    d_v0_data.resize(xRes * (yRes + 1) * zRes);
    d_w0_data.resize(xRes * yRes * (zRes + 1));

    d_u_data.resize((xRes + 1) * yRes * zRes);
    d_v_data.resize(xRes * (yRes + 1) * zRes);
    d_w_data.resize(xRes * yRes * (zRes + 1));

    // u
    OPENMP_FOR_COLLAPSE
    FOR_EACH_FACE_X
    {
        int offset = i + j * (xRes + 1) + k * (xRes + 1) * yRes;
        h_u0_scanline[offset] = 0.0f;
    }

    // v
    OPENMP_FOR_COLLAPSE
    FOR_EACH_FACE_Y
    {
        int offset = i + (j + k * (yRes + 1)) * xRes;
        h_v0_scanline[offset] = 0.0f;
        int dx = i - SOURCE_CENTER_X;
        int dz = k - SOURCE_CENTER_Z;
        int d2xz = dx * dx + dz * dz;
        if(j >= yRes - SOURCE_SIZE_Y && d2xz < SOURCE_RADIUS_XZ * SOURCE_RADIUS_XZ)
        {
            h_v0_scanline[offset] = INIT_VELOCITY;
        }
    }

        // w
    OPENMP_FOR_COLLAPSE
        FOR_EACH_FACE_Z
    {
        int offset = i + (j + k * yRes) * xRes;
        h_w0_scanline[offset] = 0.0f;
    }

    d_u0_data = h_u0_scanline;
    d_v0_data = h_v0_scanline;
    d_w0_data = h_w0_scanline;

    u0.data = d_u0_data.data();
    v0.data = d_v0_data.data();
    w0.data = d_w0_data.data();

    u.data = d_u_data.data();
    v.data = d_v_data.data();
    w.data = d_w_data.data();
}

void Smoke::initDensity()
{
    thrust::host_vector<float> h_density0_scanline(xRes * yRes * zRes);

    d_density0_data.resize(xRes * yRes * zRes);
    d_density_data.resize(xRes * yRes * zRes);

    OPENMP_FOR_COLLAPSE
        FOR_EACH_CELL
    {
        int offset = i + (j + k * yRes) * xRes;
        int dx = i - SOURCE_CENTER_X;
        int dz = k - SOURCE_CENTER_Z;
        int d2xz = dx * dx + dz * dz;
        if(j >= yRes - SOURCE_SIZE_Y && d2xz < SOURCE_RADIUS_XZ * SOURCE_RADIUS_XZ)
        {
            h_density0_scanline[offset] = INIT_DENSITY;
        }
    }

    d_density0_data = h_density0_scanline;

    density0.data = d_density0_data.data();
    density.data = d_density_data.data();
}

void Smoke::initTemperature()
{
    thrust::host_vector<float> h_temperature0_scanline(xRes * yRes * zRes);

    d_temperature0_data.resize(xRes * yRes * zRes);
    d_temperature_data.resize(xRes * yRes * zRes);

    OPENMP_FOR_COLLAPSE
        FOR_EACH_CELL
    {
        int offset = i + (j + k * yRes) * xRes;
        int dx = i - SOURCE_CENTER_X;
        int dz = k - SOURCE_CENTER_Z;
        int d2xz = dx * dx + dz * dz;
        h_temperature0_scanline[offset] = 0.0f;
        if(j >= yRes - SOURCE_SIZE_Y && d2xz < SOURCE_RADIUS_XZ * SOURCE_RADIUS_XZ)
        {
            h_temperature0_scanline[offset] = INIT_TEMPERATURE;
        }
    }

    d_temperature0_data = h_temperature0_scanline;

    temperature0.data = d_temperature0_data.data();
    temperature.data = d_temperature_data.data();
}

void Smoke::setObstacles()
{
    std::vector<char> h_scanline_obstacles(xRes * yRes * zRes);
    d_obstacles_data.resize(xRes * yRes * zRes);

    std::string filename = "resources/fan2_" + std::to_string(DIM) + ".sdf";
    std::ifstream fin( filename.c_str(), std::ios::in | std::ios::binary );
  
    fin.read(h_scanline_obstacles.data(), sizeof(char) * xRes * yRes * zRes);
    fin.close();

    d_obstacles_data = h_scanline_obstacles;

    obstacles.data = d_obstacles_data.data();
}


