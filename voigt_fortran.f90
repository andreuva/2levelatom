      !> Computes the Voigt and Faraday-Voigt profiles\n
      !!    v(dfloat): Normalized frequency\n
      !!    a(dfloat): Normalized damping\n
      !!   t(dcomplx): Complex Voigt profile
      subroutine voigt(v,a,t)

      ! I/O

      double precision, intent(in):: v, a
      complex(kind=8):: t

      ! Local

      real:: sa, sv, s, d

      complex(kind=8):: z,u,nt,dt,x,y

      sa = sngl(a)
      sv = sngl(v)

      s = abs(sv)+sa
      d = .195e0*abs(sv)-.176e0

      z = dcmplx(a,-v)

      if(s.ge..15e2) then
        t = .5641896d0*z/(.5d0+z*z)
      else

        if(s.ge..55e1) then
          u = z*z
          t = z*(.1410474d1 + .5641896d0*u)/(.75d0 + u*(.3d1 + u))
        else

          if(sa.ge.d) then

            nt = .164955d2 + z*(.2020933d2 + z*(.1196482d2 + &
                 z*(.3778987d1 + .5642236d0*z)))

            dt = .164955d2 + z*(.3882363d2 + z*(.3927121d2 + &
                 z*(.2169274d2 + z*(.6699398d1 + z))))

            t = nt/dt

          else

            u = z*z

            x = z*(.3618331d5 - u*(.33219905d4 - u*(.1540787d4 - &
                u*(.2190313d3 - u*(.3576683d2 - u*(.1320522d1 - &
                .56419d0*u))))))

            y = .320666d5 - u*(.2432284d5 - u*(.9022228d4 - &
                u*(.2186181d4 - u*(.3642191d3 - u*(.6157037d2 - &
                u*(.1841439d1 - u))))))

            t = exp(u) - x/y

          end if
        end if
      end if

      end subroutine voigt
