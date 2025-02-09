      subroutine umat(stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt,
     & drplde, stran, dstran, time, dtime, temp, dtemp, predef,
     & dpred, cmname, ndi, nshr, ntens, nstatv, props, nprops,
     & coords, drot, pnewdt, celent, dfgrd0, dfgrd1, noel, npt,
     & layer, kspt, jstep, kinc)
C
C     This UMAT is a “bridge” that sends the current strain and material
C     properties (assumed to be props(1)=E and props(2)=nu) via MPI to a
C     Python process which computes the elastic stress and consistent tangent.
C
      implicit none
C
C     Abaqus UMAT arguments
      integer ndi, nshr, ntens, nstatv, nprops, noel, npt, layer, kspt, jstep, kinc
      character*80 cmname
      double precision stress(ntens), statev(nstatv), ddsdde(ntens,ntens)
      double precision sse, spd, scd, rpl, ddsddt, drplde, dtime, temp, dtemp
      double precision predef(*), dpred(*)
      double precision stran(ntens), dstran(ntens)
      double precision time(2)
      double precision props(nprops)
      double precision coords(3), drot(3,3), pnewdt, celent
      double precision dfgrd0(3,3), dfgrd1(3,3)
C
C     MPI variables
      integer ierr, myrank, dest, tag_send, tag_recv, count_send, count_recv
      integer status(MPI_STATUS_SIZE)
      double precision send_data(8)
      double precision recv_data(42)
      integer i, j, k
C
      include 'mpif.h'
C
C     (Assume that MPI has already been initialized by Abaqus.)
      call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
C
C     In this simple example we assume that the Python process is running
C     on rank 1.
      dest = 1
      tag_send = 100
      tag_recv = 101
C
C     Pack the data to send:
C       - The first 6 entries are the strain (assumed to be in order 1–6)
C       - The next two are the material properties: E = props(1) and nu = props(2)
      do i = 1, 6
         send_data(i) = stran(i)
      end do
      send_data(7) = props(1)
      send_data(8) = props(2)
      count_send = 8
C
C     Send the data array to the Python process
      call MPI_Send(send_data, count_send, MPI_DOUBLE_PRECISION, dest, tag_send, MPI_COMM_WORLD, ierr)
C
C     Now receive the computed results.
C     We expect 6 stress components plus 36 tangent components = 42 numbers.
      count_recv = 42
      call MPI_Recv(recv_data, count_recv, MPI_DOUBLE_PRECISION, dest, tag_recv, MPI_COMM_WORLD, status, ierr)
C
C     Unpack the received stress vector
      do i = 1, 6
         stress(i) = recv_data(i)
      end do
C
C     Unpack the tangent matrix.
C     (Here we assume that the Python program sends the 6×6 tangent in row‐major order.
C      Since Fortran stores arrays in column‐major order, the simplest (but not unique)
C      strategy is to read the numbers in the same order into ddsdde(i,j) as shown below.)
      k = 7
      do j = 1, 6
         do i = 1, 6
            ddsdde(i,j) = recv_data(k)
            k = k + 1
         end do
      end do
C
C     For this elastic example, simply set the new time step size equal to dtime.
      pnewdt = dtime
C
      return
      end
